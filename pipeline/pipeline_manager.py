"""
Pipeline manager module for DOM formula assignment pipeline.
Orchestrates the entire pipeline workflow.
"""

import os
import pandas as pd
from pipeline.logger import get_logger
from pipeline.data_loader import DataLoader
from pipeline.model_trainer import ModelTrainer
from pipeline.predictor import Predictor
from pipeline.evaluator import Evaluator
from pipeline.config import ConfigManager, PipelineConfig
from pipeline.utils.utils import ensure_dir
from pipeline import plotting


class PipelineManager:
    """Orchestrates the entire DOM formula assignment pipeline."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.data_loader = DataLoader()
        self.evaluator = Evaluator()
        self.logger = get_logger("PipelineManager")
    
    def run_single_pipeline(self, config: PipelineConfig, force_retrain=False):
        """
        Run pipeline for a single model configuration.
        
        Args:
            config: PipelineConfig - pipeline configuration
            force_retrain: bool - whether to retrain existing models
            
        Returns:
            pd.DataFrame: evaluation statistics
        """
        self.logger.info(f"Running pipeline for {config.version_name}")
        
        # Ensure result directory exists
        ensure_dir(config.result_dir)
        
        # Initialize components
        trainer = ModelTrainer(
            k_neighbors=config.k_neighbors,
            metric=config.metric,
            p=config.p
        )
        
        try:
            # Load training data
            all_training_sources = config.get_all_training_sources(self.config_manager.data_config)
            self.logger.info(f"Training sources: {all_training_sources}")
            
            # Check if we should use ensemble mode
            if config.use_ensemble and len(all_training_sources) > 1:
                self.logger.info(f"Using ensemble mode with {len(all_training_sources)} separate models")
                
                # Load training data separately for each source
                separate_training_data = self.data_loader.load_training_data_separate(all_training_sources)
                
                # Generate model paths for each training source
                model_paths = config.get_model_paths_for_ensemble(all_training_sources)
                
                # Train and save multiple models
                self.logger.info("Training ensemble models...")
                trainer.train_and_save_multiple(separate_training_data, model_paths, force_retrain=force_retrain)
                
                # Load all models for predictions
                models = trainer.load_multiple_models(model_paths)
                predictor = Predictor(models=models)
                
            else:
                self.logger.info("Using single model mode")
                
                # Load combined training data
                training_data = self.data_loader.load_training_data(all_training_sources)
                
                # Train and save single model
                self.logger.info("Training model...")
                trainer.train_and_save(training_data, config.model_path, force_retrain=force_retrain)
                
                # Load model for predictions
                model = trainer.load_model(config.model_path)
                predictor = Predictor(model=model)
            
            # Load test data and make predictions (using custom test files or folder)
            self.logger.info("Loading test data and making predictions...")
            test_source = config.get_test_files_or_folder(self.config_manager.data_config)
            test_data = self.data_loader.load_testing_data(test_source)
            stat_summary = predictor.predict_testset(test_data, config.result_dir)
            
            # Load peak list data and make predictions
            self.logger.info("Processing peak lists...")
            peaklist_folder = config.get_peaklist_folder(self.config_manager.data_config)
            peaklist_files = self.data_loader.load_peaklist_data(peaklist_folder)
            predictor.predict_peaklist(peaklist_files, config.result_dir)
            
            # Save evaluation summary
            stat_df = self.evaluator.save_evaluation_summary(
                stat_summary, config.result_dir, config.version_name
            )
            
            self.logger.info(f"Pipeline completed for {config.version_name}")
            return stat_df
            
        except Exception as e:
            self.logger.error(f"Pipeline failed for {config.version_name}: {e}")
            raise
    
    def run_all_pipelines(self, force_retrain=False, plots_only=False):
        """
        Run pipelines for all standard model configurations.
        
        Args:
            force_retrain: bool - whether to retrain existing models
            plots_only: bool - whether to only generate plots from existing results
            
        Returns:
            dict: results for each model version
        """
        # Get all standard configurations
        configs = self.config_manager.get_standard_configs()
        results = {}

        if not plots_only:
            self.logger.info("Starting full pipeline run...")
            
            # Run each pipeline
            for config in configs:
                try:
                    stat_df = self.run_single_pipeline(config, force_retrain=force_retrain)
                    results[config.version_name] = stat_df
                except Exception as e:
                    self.logger.error(f"Failed to run pipeline for {config.version_name}: {e}")
                    results[config.version_name] = pd.DataFrame()
        else:
            self.logger.info("Plots-only mode: loading existing results...")
            
            # Load existing results from output directories
            for config in configs:
                try:
                    results[config.version_name] = self._load_existing_results(config)
                    print(f"Loaded results for {config.version_name}, {len(results[config.version_name])} entries")
                except Exception as e:
                    self.logger.warning(f"Could not load existing results for {config.version_name}: {e}")
                    results[config.version_name] = pd.DataFrame()
            
        # Generate plots
        self._generate_plots(results)
        
        # Save combined comparison
        self._save_combined_comparison(results)
        
        self.logger.info("All pipelines completed")
        return results
    
    def _generate_plots(self, results):
        """
        Generate visualization plots for all results.
        
        Args:
            results: dict - results from all pipeline runs
        """
        self.logger.info("Generating plots...")
        
        try:
            # Prepare data for plotting
            result_dirs = []
            labels = []
            
            # Get all configurations to check for existing result directories
            all_configs = self.config_manager.get_standard_configs()
            config_by_name = {config.version_name: config for config in all_configs}
            
            for version_name in results.keys():
                if version_name in config_by_name:
                    config = config_by_name[version_name]
                    result_dir = config.result_dir
                    
                    # Check if result directory exists and has any result files
                    if self._has_plot_data(result_dir):
                        result_dirs.append(result_dir)
                        labels.append(version_name)
                        self.logger.info(f"Including {version_name} in plots (result dir: {result_dir})")
                    else:
                        self.logger.warning(f"No plot data found for {version_name} in {result_dir}")
            
            if result_dirs:
                # Generate testset plots
                plotting.plot_testset_main(
                    result_dirs, labels, self.config_manager.data_config.output_dir
                )
                
                # Generate peaklist plots
                plotting.plot_peaklist_main(
                    result_dirs, labels, self.config_manager.data_config.output_dir
                )
                
                self.logger.info("Plots generated successfully")
            else:
                self.logger.warning("No valid results found for plotting")
                
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
    
    def _save_combined_comparison(self, results):
        """
        Save combined comparison of all model results.
        
        Args:
            results: dict - results from all pipeline runs
        """
        try:
            # Combine all statistics
            all_stats = []
            for version_name, stat_df in results.items():
                if not stat_df.empty:
                    all_stats.append(stat_df)
            
            if all_stats:
                combined_stats = pd.concat(all_stats, ignore_index=True)
                comparison_path = os.path.join(
                    self.config_manager.data_config.base_dir, 
                    "knn_comparison_summary.csv"
                )
                combined_stats.to_csv(comparison_path, index=False)
                self.logger.info(f"Combined comparison saved to: {comparison_path}")
            else:
                self.logger.warning("No valid statistics found for comparison")
                
        except Exception as e:
            self.logger.error(f"Error saving combined comparison: {e}")
    
    def _has_plot_data(self, result_dir):
        """
        Check if a result directory has data that can be plotted.
        
        Args:
            result_dir: str - path to result directory
            
        Returns:
            bool: True if directory has plottable data
        """
        if not os.path.exists(result_dir):
            return False
            
        # Check for test result files (results_*.csv)
        has_test_results = False
        try:
            for file in os.listdir(result_dir):
                if file.startswith('results_') and file.endswith('.csv'):
                    has_test_results = True
                    break
        except (OSError, PermissionError):
            pass
        
        # Check for peaklist results
        has_peaklist_results = False
        peaklist_dir = os.path.join(result_dir, 'peak_list')
        if os.path.exists(peaklist_dir):
            try:
                for file in os.listdir(peaklist_dir):
                    if file.endswith('.csv'):
                        has_peaklist_results = True
                        break
            except (OSError, PermissionError):
                pass
        
        return has_test_results or has_peaklist_results

    def _load_existing_results(self, config):
        """
        Load existing evaluation results from output directory.
        
        Args:
            config: PipelineConfig - pipeline configuration
            
        Returns:
            pd.DataFrame: existing evaluation statistics or empty DataFrame
        """
        try:
            # Look for evaluation summary file in the result directory
            # Try the standard filename first
            eval_file = os.path.join(config.result_dir, "evaluation_summary_stats.csv")
            
            if os.path.exists(eval_file):
                stat_df = pd.read_csv(eval_file)
                self.logger.info(f"Loaded existing results for {config.version_name} from {eval_file}")
                return stat_df
            else:
                # Try alternative naming pattern
                alt_eval_file = os.path.join(config.result_dir, f"{config.version_name}_evaluation_summary.csv")
                if os.path.exists(alt_eval_file):
                    stat_df = pd.read_csv(alt_eval_file)
                    self.logger.info(f"Loaded existing results for {config.version_name} from {alt_eval_file}")
                    return stat_df
                else:
                    self.logger.warning(f"No existing evaluation file found at {eval_file} or {alt_eval_file}")
                    return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error loading existing results for {config.version_name}: {e}")
            return pd.DataFrame()
    
    def run_custom_pipeline(self, version_name, training_folders, model_path, 
                          result_dir, k_neighbors=1, metric='minkowski', p=2, 
                          testing_folder=None, peaklist_folder=None, custom_test_files=None, 
                          force_retrain=False, use_ensemble=True):
        """
        Run pipeline with custom configuration.
        
        Args:
            version_name: str - name for this pipeline run
            training_folders: str or list - training data folder(s)
            model_path: str - path to save/load model
            result_dir: str - directory for results
            k_neighbors: int - number of neighbors for KNN
            metric: str - distance metric for KNN
            p: int - power parameter for Minkowski metric
            testing_folder: str - custom testing folder (optional)
            peaklist_folder: str - custom peaklist folder (optional)
            custom_test_files: list - specific test files to use (optional)
            force_retrain: bool - whether to retrain existing models
            use_ensemble: bool - whether to train separate models per dataset (default True)
            
        Returns:
            pd.DataFrame: evaluation statistics
        """
        # Create custom configuration
        config = self.config_manager.create_custom_config(
            version_name=version_name,
            training_folders=training_folders,
            model_path=model_path,
            result_dir=result_dir,
            k_neighbors=k_neighbors,
            metric=metric,
            p=p,
            testing_folder=testing_folder,
            peaklist_folder=peaklist_folder,
            custom_test_files=custom_test_files,
            use_ensemble=use_ensemble
        )
        
        # Run pipeline
        return self.run_single_pipeline(config, force_retrain=force_retrain)


# Legacy function for backward compatibility
def run_pipeline_from_folder(version_name, training_folder, testing_folder, 
                           model_path, result_dir, k=1):
    """
    Legacy function for backward compatibility.
    
    Args:
        version_name: str - model version name
        training_folder: str or list - training data folder(s)
        testing_folder: str - testing data folder
        model_path: str - path to save/load model
        result_dir: str - directory for results
        k: int - number of neighbors
        
    Returns:
        pd.DataFrame: evaluation statistics
    """
    manager = PipelineManager()
    
    # Update testing folder in config
    manager.config_manager.update_data_config(testing_folder=testing_folder)
    
    # Run custom pipeline
    return manager.run_custom_pipeline(
        version_name=version_name,
        training_folders=training_folder,
        model_path=model_path,
        result_dir=result_dir,
        k_neighbors=k
    )


def run_main(plots_only=False):
    """
    Main function to run all pipelines.
    
    Args:
        plots_only: bool - if True, skip pipeline execution and only generate plots from existing results
    """
    manager = PipelineManager()
    results = manager.run_all_pipelines(plots_only=plots_only)
    return results