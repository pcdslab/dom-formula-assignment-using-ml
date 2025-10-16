"""
Predictor module for DOM formula assignment pipeline.
Handles predictions on peak lists and test data.
"""

import os
import pandas as pd
from pipeline.logger import get_logger
from pipeline.utils.utils import calculate_theoretical_mass, ensure_dir


class Predictor:
    """Handles predictions using trained KNN model(s)."""
    
    def __init__(self, model=None, models=None):
        """
        Initialize predictor with either a single model or multiple models for ensemble.
        
        Args:
            model: single KNN model (for backward compatibility)
            models: list of tuples [(source_name, model), ...] for ensemble predictions
        """
        self.model = model
        self.models = models if models else []
        self.use_ensemble = len(self.models) > 1
        self.logger = get_logger("Predictor")
        
        if self.use_ensemble:
            self.logger.info(f"Initialized ensemble predictor with {len(self.models)} models")
        elif self.model:
            self.logger.info("Initialized single-model predictor")
        else:
            self.logger.warning("Initialized predictor without any models")
    
    def predict_peaklist(self, peaklist_files, result_dir):
        """
        Make predictions on peak list files.
        
        Args:
            peaklist_files: list - list of tuples (filepath, filename)
            result_dir: str - directory to save results
        """
        peaklist_output_dir = os.path.join(result_dir, "peak_list")
        ensure_dir(peaklist_output_dir)
        
        self.logger.info(f"Predicting on {len(peaklist_files)} peak list files")
        
        for file_path, filename in peaklist_files:
            self.logger.info(f"Processing peak list: {filename}")
            
            try:
                # Load peak list data
                data = pd.read_csv(file_path)
                
                # Standardize column names
                if 'm/z Exp.' in data.columns:
                    data.rename(columns={'m/z Exp.': 'm/z exp.'}, inplace=True)
                
                # Make predictions
                predictions = []
                for _, row in data.iterrows():
                    prediction = self._predict_single_peak(row)
                    predictions.append(prediction)
                
                # Save results
                df = pd.DataFrame(predictions)
                output_path = os.path.join(peaklist_output_dir, filename)
                df.to_csv(output_path, index=False)
                
                valid_count = df['valid_prediction'].sum()
                self.logger.info(f"Peak list predictions saved to: {output_path} "
                               f"(Valid predictions: {valid_count}/{len(df)})")
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
    
    def predict_testset(self, test_data, result_dir):
        """
        Make predictions on test data files.
        
        Args:
            test_data: list - list of tuples (filename, dataframe)
            result_dir: str - directory to save results
            
        Returns:
            list: list of prediction statistics for each file
        """
        ensure_dir(result_dir)
        stat_summary = []
        
        self.logger.info(f"Evaluating model on {len(test_data)} test files")
        
        for filename, df in test_data:
            self.logger.info(f"Processing test file: {filename}")
            
            try:
                predictions = []
                for _, row in df.iterrows():
                    prediction = self._predict_test_sample(row)
                    predictions.append(prediction)
                
                # Create results dataframe
                pred_df = pd.DataFrame(predictions)
                
                # Calculate statistics
                stats = self._calculate_prediction_stats(pred_df, filename)
                stat_summary.append(stats)
                
                # Save results
                results_file = os.path.join(result_dir, f"results_{filename.replace('.xlsx', '')}.csv")
                pred_df.to_csv(results_file, index=False)
                
                self.logger.info(f"Results saved to: {results_file} | "
                               f"True: {stats['True Predictions']}, "
                               f"False: {stats['False Predictions']}, "
                               f"No: {stats['No Predictions']}")
                
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
        
        return stat_summary
    
    def _predict_single_peak(self, row):
        """
        Make prediction for a single peak in peak list.
        Uses ensemble prediction if multiple models are available.
        
        Args:
            row: pandas Series - peak data row
            
        Returns:
            dict: prediction results
        """
        mz = round(row['m/z exp.'], 5)
        intensity = row["Intensity"]
        
        if self.use_ensemble:
            # Ensemble prediction: query all models and choose best by PPM error
            best_prediction = None
            best_ppm_error = float('inf')
            
            for source_name, model in self.models:
                # Make prediction
                pred_formula = model.predict([[mz]])[0]
                
                # Get training m/z value from nearest neighbor
                knn_indices = model.kneighbors([[mz]], n_neighbors=1, return_distance=False)
                training_mz = model._fit_X[knn_indices][0][0][0]
                
                # Calculate mass and error
                pred_formula_mass = round(calculate_theoretical_mass(pred_formula), 5)
                pred_formula_mass_error = round(abs(pred_formula_mass - mz), 4)
                mass_error_ppm = (abs(pred_formula_mass - mz) / mz) * (10**6)
                
                # Check if this is the best prediction so far
                if mass_error_ppm < best_ppm_error:
                    best_ppm_error = mass_error_ppm
                    valid_prediction = 1 if mass_error_ppm <= 1 else 0
                    
                    best_prediction = {
                        'm/z exp.': mz,
                        'Intensity': intensity,
                        'predicted_formula': pred_formula,
                        'pred_formula_mass': pred_formula_mass,
                        'training_mz': training_mz,
                        'pred_formula_mass_error': pred_formula_mass_error,
                        'mass_error_in_ppm': mass_error_ppm,
                        'valid_prediction': valid_prediction,
                        'model_source': source_name
                    }
            
            return best_prediction
        else:
            # Single model prediction
            model = self.model if self.model else self.models[0][1]
            
            # Make prediction
            pred_formula = model.predict([[mz]])[0]
            
            # Get training m/z value from nearest neighbor
            knn_indices = model.kneighbors([[mz]], n_neighbors=1, return_distance=False)
            training_mz = model._fit_X[knn_indices][0][0][0]
            
            # Calculate mass and error
            pred_formula_mass = round(calculate_theoretical_mass(pred_formula), 5)
            pred_formula_mass_error = round(abs(pred_formula_mass - mz), 4)
            mass_error_ppm = (abs(pred_formula_mass - mz) / mz) * (10**6)
            valid_prediction = 1 if mass_error_ppm <= 1 else 0
            
            result = {
                'm/z exp.': mz,
                'Intensity': intensity,
                'predicted_formula': pred_formula,
                'pred_formula_mass': pred_formula_mass,
                'training_mz': training_mz,
                'pred_formula_mass_error': pred_formula_mass_error,
                'mass_error_in_ppm': mass_error_ppm,
                'valid_prediction': valid_prediction
            }
            
            if self.use_ensemble:
                result['model_source'] = self.models[0][0]
            
            return result
    
    def _predict_test_sample(self, row):
        """
        Make prediction for a single test sample.
        Uses ensemble prediction if multiple models are available.
        
        Args:
            row: pandas Series - test data row
            
        Returns:
            dict: prediction results with evaluation metrics
        """
        mz = round(row['m/z exp.'], 5)
        true_formula = row['Chem. Formula']
        
        if self.use_ensemble:
            # Ensemble prediction: query all models and choose best by PPM error
            best_prediction = None
            best_ppm_error = float('inf')
            
            for source_name, model in self.models:
                # Make prediction
                pred_formula = model.predict([[mz]])[0]
                
                # Calculate masses and errors
                true_formula_mass = calculate_theoretical_mass(true_formula)
                pred_formula_mass = calculate_theoretical_mass(pred_formula)
                true_formula_mass_error = round(abs(true_formula_mass - mz), 4)
                pred_formula_mass_error = round(abs(pred_formula_mass - mz), 4)
                mass_error_ppm = (abs(pred_formula_mass - mz) / mz) * (10**6)
                
                # Check if this is the best prediction so far
                if mass_error_ppm < best_ppm_error:
                    best_ppm_error = mass_error_ppm
                    
                    # Calculate prediction categories
                    true_prediction = 1 if pred_formula == true_formula else 0
                    false_prediction = 1 if (pred_formula != true_formula and mass_error_ppm < 1) else 0
                    no_prediction = 1 if (pred_formula != true_formula and mass_error_ppm >= 1) else 0
                    
                    best_prediction = {
                        'm/z exp.': mz,
                        'proposed_formula': true_formula,
                        'predicted_formula': pred_formula,
                        'true_prediction': true_prediction,
                        'false_prediction': false_prediction,
                        'no_prediction': no_prediction,
                        'proposed_formula_mass': true_formula_mass,
                        'pred_formula_mass': pred_formula_mass,
                        'proposed_formula_mass_error': true_formula_mass_error,
                        'pred_formula_mass_error': pred_formula_mass_error,
                        'mass_error_in_ppm': mass_error_ppm,
                        'model_source': source_name
                    }
            
            return best_prediction
        else:
            # Single model prediction
            model = self.model if self.model else self.models[0][1]
            
            # Make prediction
            pred_formula = model.predict([[mz]])[0]
            
            # Calculate masses and errors
            true_formula_mass = calculate_theoretical_mass(true_formula)
            pred_formula_mass = calculate_theoretical_mass(pred_formula)
            true_formula_mass_error = round(abs(true_formula_mass - mz), 4)
            pred_formula_mass_error = round(abs(pred_formula_mass - mz), 4)
            mass_error_ppm = (abs(pred_formula_mass - mz) / mz) * (10**6)
            
            # Calculate prediction categories
            true_prediction = 1 if pred_formula == true_formula else 0
            false_prediction = 1 if (pred_formula != true_formula and mass_error_ppm < 1) else 0
            no_prediction = 1 if (pred_formula != true_formula and mass_error_ppm >= 1) else 0
            
            result = {
                'm/z exp.': mz,
                'proposed_formula': true_formula,
                'predicted_formula': pred_formula,
                'true_prediction': true_prediction,
                'false_prediction': false_prediction,
                'no_prediction': no_prediction,
                'proposed_formula_mass': true_formula_mass,
                'pred_formula_mass': pred_formula_mass,
                'proposed_formula_mass_error': true_formula_mass_error,
                'pred_formula_mass_error': pred_formula_mass_error,
                'mass_error_in_ppm': mass_error_ppm
            }
            
            if self.use_ensemble:
                result['model_source'] = self.models[0][0]
            
            return result
    
    def _calculate_prediction_stats(self, pred_df, filename):
        """
        Calculate prediction statistics for a test file.
        
        Args:
            pred_df: pd.DataFrame - predictions dataframe
            filename: str - test file name
            
        Returns:
            dict: prediction statistics
        """
        return {
            'Filename': filename,
            'Total Count': len(pred_df),
            'True Predictions': pred_df['true_prediction'].sum(),
            'False Predictions': pred_df['false_prediction'].sum(),
            'No Predictions': pred_df['no_prediction'].sum()
        }