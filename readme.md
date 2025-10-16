# DOM Formula Assignment Pipeline

A machine learning pipeline for assigning molecular formulas to dissolved organic matter (DOM) mass spectrometry data using K-Nearest Neighbors (KNN) models.

##  Overview

This research implements an automated pipeline for formula assignment in DOM (Dissolved Organic Matter) mass spectrometry analysis. It uses KNN classification models to predict molecular formulas based on mass-to-charge ratios, supporting multiple training datasets, distance metrics, and K-neighbor configurations.

### Key Features

- **Modular Architecture**: Separate components for data loading, model training, prediction, and evaluation
- **Multiple Distance Metrics**: Support for Euclidean, Manhattan, Cosine, and other distance metrics
- **Ensemble Learning**: Train separate models per dataset for improved predictions
- **Flexible Data Sources**: Mix training data from folders and individual files
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, and assignment rates
- **Rich Visualizations**: Automated generation of comparison plots, confusion matrices, and word clouds
- **Configurable Pipelines**: Easy configuration management for different experimental setups

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, wordcloud, joblib

### Basic Usage

```python
from pipeline import run_main

# Run all standard pipelines (7T, 21T, Combined, Synthetic)
results = run_main()

# Generate plots only from existing results (skip training/prediction)
results = run_main(plots_only=True)
```

### Running from Command Line

```bash
python run_pipeline.py
```

## Architecture Overview

The pipeline has been refactored into separate, focused components for better maintainability, testability, and extensibility.

## Pipeline Components

### 1. `data_loader.py` - DataLoader
**Purpose**: Handles loading and preprocessing of training and testing data

**Key Methods**:
- `load_training_data(training_sources)`: Load training data from Excel/CSV files or folders
- `load_training_data_separate(training_sources)`: Load data separately for ensemble models
- `load_testing_data(testing_folder)`: Load testing data from Excel files
- `load_peaklist_data(peak_list_dir)`: Load peak list data from CSV files

**Features**:
- Supports both single files and folders
- Mix folders and individual files in training sources
- Automatic column name standardization
- Data validation and cleaning
- Comprehensive error handling and logging

### 2. `model_trainer.py` - ModelTrainer
**Purpose**: Handles KNN model training and model management

**Key Methods**:
- `train_model(training_data)`: Train KNN model on provided data
- `train_and_save_multiple(training_data_list, model_paths)`: Train ensemble models
- `save_model(model_path)`: Save trained model to disk
- `load_model(model_path)`: Load trained model from disk
- `load_multiple_models(model_paths)`: Load ensemble models

**Features**:
- Configurable k-neighbors parameter (K=1, 3, 5, etc.)
- Multiple distance metrics (Euclidean, Manhattan, Cosine, Chebyshev)
- Model persistence with joblib
- Ensemble model support
- Skip training if model already exists (optional override)

### 3. `predictor.py` - Predictor
**Purpose**: Handles predictions on peak lists and test data

**Key Methods**:
- `predict_peaklist(peaklist_files, result_dir)`: Make predictions on peak list files
- `predict_testset(test_data, result_dir)`: Make predictions on test data

**Features**:
- Mass error calculation in PPM
- Validation of predictions (≤1 PPM error threshold)
- Nearest neighbor information retrieval
- Support for single model or ensemble predictions
- Structured output with comprehensive metrics

### 4. `evaluator.py` - Evaluator
**Purpose**: Handles model evaluation and metrics calculation

**Key Methods**:
- `save_evaluation_summary(stat_summary, result_dir, model_version)`: Save evaluation statistics
- `calculate_accuracy_metrics(stat_df)`: Calculate accuracy, precision, recall metrics
- `compare_models(stat_dfs, model_names)`: Compare multiple models
- `generate_evaluation_report(result_dir, model_version)`: Generate comprehensive report

**Features**:
- Multiple evaluation metrics (accuracy, precision, recall, assignment rate)
- Model comparison functionality
- Summary statistics across all test files
- CSV export of results

### 5. `config.py` - Configuration Management
**Purpose**: Centralized configuration management

**Key Classes**:
- `ModelConfig`: Model-specific parameters (k_neighbors, weights, metric, p)
- `DataConfig`: Data paths and folder configurations
- `PipelineConfig`: Individual pipeline run configuration
- `ConfigManager`: Manages all configurations

**Features**:
- Dataclass-based configuration
- Standard configuration presets (7T, 21T, Combined, Synthetic)
- Custom configuration creation
- Dynamic configuration updates
- Support for different distance metrics and K values

### 6. `pipeline_manager.py` - PipelineManager
**Purpose**: Orchestrates the entire pipeline workflow

**Key Methods**:
- `run_single_pipeline(config)`: Run pipeline for one model configuration
- `run_all_pipelines()`: Run all standard model configurations
- `run_custom_pipeline(...)`: Run pipeline with custom parameters

**Features**:
- Complete workflow orchestration
- Automatic plot generation
- Combined results comparison
- Error handling and recovery
- Plots-only mode for visualization without retraining
- Ensemble and single model support

### 7. `plotting.py` - Visualization
**Purpose**: Generate comprehensive visualizations

**Key Functions**:
- `plot_testset_main(result_dirs, labels, out_dir)`: Main testset plotting
- `plot_peaklist_main(result_dirs, labels, out_dir)`: Main peaklist plotting
- `plot_wordcloud_grid(...)`: Word cloud visualizations of predicted formulas
- Various specific plotting functions for different analyses

**Generated Plots**:
- Valid predictions comparison across models
- Assignment rate comparisons
- Formula distribution word clouds
- Per-file performance metrics
- Confusion matrices
- Model comparison charts

### 8. `utils/` - Utility Functions
**Purpose**: Helper functions for the pipeline

**Key Modules**:
- `utils.py`: Common utility functions (mass calculations, directory management)
- `generate_formula_combinations.py`: Synthetic data generation for testing

## Data Structure

### Training Data
Located in `data/` directory:
- **DOM_training_set_ver2/**: 7T mass spectrometry data (8 files)
  - Harney River samples (5 files)
  - Pantanal samples (2 files)
  - Suwannee River Fulvic Acid (1 file)
- **DOM_training_set_ver3/**: 21T mass spectrometry data
- **synthetic_data/**: Synthetically generated formula-mass pairs

### Testing Data
- **DOM_testing_set/**: Test files for model validation
  - Pahokee River Fulvic Acid
  - Suwannee River Fulvic Acid 2 (v2)
  - Suwannee River Fulvic Acid 3
- **DOM_testing_set_Peaklists/**: Peak list CSV files for prediction

### Data Format
**Training/Testing Files** (Excel/CSV):
- Required columns: `Mass_Daltons`, `Formula`
- Additional columns preserved in output

**Peak List Files** (CSV):
- Required columns: `m/z exp.` (or `m/z Exp.`)
- Mass values for formula prediction

## Standard Model Configurations

The pipeline includes four standard model configurations:

### 1. Model-7T
- **Training Data**: DOM_training_set_ver2 (7T data)
- **Description**: Model trained on 7 Tesla mass spectrometry data
- **Use Case**: Standard resolution mass spectrometry

### 2. Model-21T
- **Training Data**: DOM_training_set_ver3 (21T data)
- **Description**: Model trained on 21 Tesla mass spectrometry data
- **Use Case**: High resolution mass spectrometry

### 3. Model-7T-21T (Combined)
- **Training Data**: Both ver2 and ver3
- **Description**: Ensemble model combining 7T and 21T data
- **Use Case**: Leveraging both resolution levels

### 4. Model-Synthetic
- **Training Data**: Combined 7T, 21T, and synthetic data
- **Description**: Enhanced with synthetically generated formulas
- **Use Case**: Extended coverage of formula space

Each model is trained with multiple configurations:
- **K values**: 1, 3 (number of nearest neighbors)
- **Distance metrics**: Euclidean (p=2), Manhattan (p=1)

This results in 16 model variants (4 models × 2 K values × 2 metrics).

## Running Experiments

### 1. Run All Standard Pipelines

```python
from pipeline import run_main

# Train models, make predictions, generate plots
results = run_main()

# Only generate plots from existing results (no training)
results = run_main(plots_only=True)
```

This will:
1. Train KNN models for all configurations (if not already trained)
2. Make predictions on test sets and peak lists
3. Calculate evaluation metrics
4. Generate comprehensive visualizations
5. Save results to `output/` directory

### 2. Run Specific Model Configuration

```python
from pipeline import PipelineManager

manager = PipelineManager()

# Get standard configurations
configs = manager.config_manager.get_standard_configs()

# Run a specific configuration
for config in configs:
    if "Model-7T_K1_Euclidean" in config.version_name:
        results = manager.run_single_pipeline(config)
        break
```

### 3. Custom Training Data Configuration

```python
from pipeline import PipelineManager

manager = PipelineManager()

# Train with custom data sources (mix of folders and files)
results = manager.run_custom_pipeline(
    version_name="Custom_Model",
    training_folders=[
        "data/DOM_training_set_ver2",  # Full folder
        "data/DOM_training_set_ver3/Table_Harney_River_1_21T.xlsx",  # Specific file
        "data/custom_training/curated_formulas.csv"  # Additional file
    ],
    model_path="models/custom_model.joblib",
    result_dir="output/custom/test_results",
    k_neighbors=3,
    metric='minkowski',
    p=2  # Euclidean distance
)
```

### 4. Test on Specific Files

```python
# Test on specific files instead of entire test folder
results = manager.run_custom_pipeline(
    version_name="Targeted_Test",
    training_folders="data/DOM_training_set_ver2",
    model_path="models/targeted_model.joblib",
    result_dir="output/targeted/test_results",
    custom_test_files=[
        "data/DOM_testing_set/Table_Pahokee_River_Fulvic_Acid.xlsx"
    ]
)
```

### 5. Ensemble vs Single Model

```python
# Ensemble mode: Train separate models for each training source
results = manager.run_custom_pipeline(
    version_name="Ensemble_Model",
    training_folders=[
        "data/DOM_training_set_ver2",
        "data/DOM_training_set_ver3"
    ],
    model_path="models/ensemble_model.joblib",
    result_dir="output/ensemble/test_results",
    use_ensemble=True  # Default
)

# Single model: Combine all training data into one model
results = manager.run_custom_pipeline(
    version_name="Single_Model",
    training_folders=[
        "data/DOM_training_set_ver2",
        "data/DOM_training_set_ver3"
    ],
    model_path="models/single_model.joblib",
    result_dir="output/single/test_results",
    use_ensemble=False
)
```

### 6. Different Distance Metrics

```python
# Manhattan distance (L1 norm)
results = manager.run_custom_pipeline(
    version_name="Manhattan_Model",
    training_folders="data/DOM_training_set_ver2",
    model_path="models/manhattan_model.joblib",
    result_dir="output/manhattan/test_results",
    k_neighbors=1,
    metric='minkowski',
    p=1  # Manhattan
)

# Cosine similarity
results = manager.run_custom_pipeline(
    version_name="Cosine_Model",
    training_folders="data/DOM_training_set_ver2",
    model_path="models/cosine_model.joblib",
    result_dir="output/cosine/test_results",
    k_neighbors=1,
    metric='cosine'
)
```

### 7. Force Model Retraining

```python
# Retrain even if model exists
results = manager.run_single_pipeline(config, force_retrain=True)
```

## Output Structure

After running the pipeline, results are organized as follows:

```
output/
├── 7T/                          # Model-7T results
│   ├── K1_Euclidean/
│   │   └── test_results/
│   │       ├── results_*.csv    # Per-file predictions
│   │       ├── evaluation_summary_stats.csv
│   │       └── peak_list/       # Peak list predictions
│   ├── K1_Manhattan/
│   ├── K3_Euclidean/
│   └── K3_Manhattan/
├── 21T/                         # Model-21T results
├── combined/                    # Model-7T-21T results
├── synthetic_combined/          # Model-Synthetic results
└── plots/                       # Visualization outputs
    ├── testset_*.png
    ├── peaklist_*.png
    └── comparison_*.png

models/                          # Trained models
├── knn_model_Model-7T_K1_Euclidean.joblib
├── knn_model_Model-7T_K1_Manhattan.joblib
├── ... (all model variants)

logs/                            # Execution logs
└── pipeline_*.log

data/
└── knn_comparison_summary.csv   # Combined comparison of all models
```

### Output Files

**Per-Test-File Results** (`results_*.csv`):
- Columns: Mass_Daltons, True_Formula, Predicted_Formula, Mass_Error_PPM, valid_prediction, nearest_neighbors, etc.

**Evaluation Summary** (`evaluation_summary_stats.csv`):
- Columns: File, Total Count, True Predictions, False Predictions, No Predictions, Assignment Rate, Model Version

**Peak List Predictions** (`peak_list/*.csv`):
- Columns: m/z exp., predicted_formula, predicted_mass, mass_error_ppm, valid_prediction, confidence

**Combined Comparison** (`knn_comparison_summary.csv`):
- Statistics for all model configurations across all test files

## Usage Examples

### Example 1: Basic Pipeline Execution
```python
from pipeline import run_main

# Run all standard pipelines (7T, 21T, Combined, Synthetic)
results = run_main()
```

### Example 2: Custom Pipeline with Mixed Data Sources
```python
from pipeline import PipelineManager

manager = PipelineManager()

# Train with combination of folders and specific files
results = manager.run_custom_pipeline(
    version_name="Custom_Experiment",
    training_folders=[
        "data/DOM_training_set_ver2",           # Base folder
        "data/additional_training",              # Additional folder
        "data/special_cases/edge_cases.xlsx",   # Specific file
        "data/validated/high_confidence.csv"    # Another file
    ],
    model_path="models/custom_experiment.joblib",
    result_dir="output/custom/test_results",
    k_neighbors=5,
    metric='minkowski',
    p=2
)
```

### Example 3: Component-Level Usage
```python
from pipeline import DataLoader, ModelTrainer, Predictor, Evaluator

# Use individual components
loader = DataLoader()
trainer = ModelTrainer(k_neighbors=3, metric='minkowski', p=2)
evaluator = Evaluator()

# Load data
training_data = loader.load_training_data("data/DOM_training_set_ver2")

# Train model  
model = trainer.train_model(training_data)
trainer.save_model("models/my_model.joblib")

# Make predictions
predictor = Predictor(model=model)
test_data = loader.load_testing_data("data/DOM_testing_set")
stats = predictor.predict_testset(test_data, "output/results")

# Evaluate
evaluator.save_evaluation_summary(stats, "output/results", "MyModel")
```

### Example 4: Global Configuration Management
```python
from pipeline import ConfigManager

config_manager = ConfigManager()

# Get standard configurations
configs = config_manager.get_standard_configs()

# Add additional training sources to ALL pipeline runs
config_manager.add_training_sources([
    "data/custom_training_1",
    "data/custom_training_2/special_samples.xlsx"
])

# Set custom test files instead of using entire folder
config_manager.set_custom_test_files([
    "data/DOM_testing_set/Table_Pahokee_River_Fulvic_Acid.xlsx"
])

# Update model parameters globally
config_manager.update_model_config(k_neighbors=5, metric='manhattan')
```

### Example 5: Targeted Testing with Additional Training
```python
from pipeline import PipelineManager

manager = PipelineManager()

# Use specific test files as additional training, then test on different file
results = manager.run_custom_pipeline(
    version_name="Enhanced_Model",
    training_folders=[
        "data/DOM_training_set_ver2",
        "data/DOM_training_set_ver3"
    ],
    additional_training_sources=[
        "data/DOM_testing_set/Table_Suwannee_River_Fulvic_Acid_2_v2.xlsx",
        "data/DOM_testing_set/Table_Suwannee_River_Fulvic_Acid_3.xlsx"
    ],
    model_path="models/enhanced_model.joblib",
    result_dir="output/enhanced/test_results",
    custom_test_files=[
        "data/DOM_testing_set/Table_Pahokee_River_Fulvic_Acid.xlsx"
    ],
    k_neighbors=3,
    use_ensemble=True
)
```

### Example 6: Comparing Different K Values and Metrics

### Example 6: Comparing Different K Values and Metrics
```python
from pipeline import PipelineManager

manager = PipelineManager()

# Test different K values
for k in [1, 3, 5, 7]:
    results = manager.run_custom_pipeline(
        version_name=f"K{k}_Comparison",
        training_folders="data/DOM_training_set_ver2",
        model_path=f"models/knn_k{k}.joblib",
        result_dir=f"output/k_comparison/K{k}/test_results",
        k_neighbors=k,
        metric='minkowski',
        p=2
    )

# Test different distance metrics
metrics = [
    ('Euclidean', 'minkowski', 2),
    ('Manhattan', 'minkowski', 1),
    ('Cosine', 'cosine', None)
]

for name, metric, p in metrics:
    params = {
        'version_name': f"{name}_Distance",
        'training_folders': "data/DOM_training_set_ver2",
        'model_path': f"models/knn_{name.lower()}.joblib",
        'result_dir': f"output/metric_comparison/{name}/test_results",
        'k_neighbors': 1,
        'metric': metric
    }
    if p is not None:
        params['p'] = p
    
    results = manager.run_custom_pipeline(**params)
```

## Advanced Features

### Training Data Flexibility

The pipeline supports ultimate flexibility in specifying training data:

**Supported Source Types**:
- **Folders**: Load all Excel/CSV files from a directory
- **Individual Files**: Specific Excel (.xlsx) or CSV (.csv) files
- **Combination**: Mix folders and files in the same configuration

**Global Additional Sources**:
```python
# Add sources that will be used by ALL pipeline runs
manager.config_manager.add_training_sources([
    "data/extra_folder",                    # folder
    "data/special/rare_compounds.xlsx",     # file
    "data/literature/validated.csv"        # file
])
```

**Per-Pipeline Custom Sources**:
```python
# Specify exact training sources for a specific pipeline
results = manager.run_custom_pipeline(
    training_folders=[
        "data/base_training",                   # folder
        "data/supplements/supplement1.xlsx",    # file
        "data/experiments/exp_data.csv",       # file
        "data/additional_folder"               # folder
    ],
    # ... other parameters
)
```

### Ensemble Learning

**How it works**:
- Each training source (folder or file) trains a separate model
- Predictions are made using all models
- Results are aggregated for final prediction

**Enable/Disable**:
```python
# Ensemble mode (default): Separate models per training source
use_ensemble=True

# Single model: Combine all training data into one model
use_ensemble=False
```

**Benefits**:
- Better handling of heterogeneous data sources
- Improved robustness
- Ability to trace predictions back to specific training sets

### Distance Metrics

**Supported Metrics**:
- **Euclidean** (Minkowski with p=2): Standard distance metric
- **Manhattan** (Minkowski with p=1): L1 norm, city-block distance
- **Cosine**: Cosine similarity
- **Chebyshev**: Maximum coordinate difference
- **Minkowski**: Generalized metric with custom p value

**Selection Criteria**:
- Euclidean: General-purpose, works well for most cases
- Manhattan: More robust to outliers, faster computation
- Cosine: Good for normalized features
- Experiment with different metrics to find optimal performance

### Prediction Validation

Predictions are validated based on mass error:
- **Valid prediction**: Mass error ≤ 1 PPM
- **Invalid prediction**: Mass error > 1 PPM
- **No prediction**: Model couldn't make a prediction

**Mass Error Calculation**:
```
Mass Error (PPM) = |Predicted Mass - Observed Mass| / Observed Mass × 10^6
```

## Evaluation Metrics

The pipeline calculates comprehensive metrics:

- **Accuracy**: (True Predictions) / (Total Samples)
- **Precision**: (True Predictions) / (True + False Predictions)
- **Recall**: Same as accuracy in this context
- **Assignment Rate**: (True + False Predictions) / (Total Samples)
- **True Predictions**: Predicted formula matches actual formula
- **False Predictions**: Predicted formula doesn't match actual formula
- **No Predictions**: Model couldn't make a valid prediction (Error > 1PPM)

## Visualization Outputs

The pipeline generates comprehensive visualizations:

### Test Set Plots
- Valid predictions comparison across models
- Assignment rate comparison
- Per-file accuracy metrics
- True/False/No predictions distribution

### Peak List Plots
- Formula distribution word clouds
- Top predicted formulas
- Prediction confidence distributions

### Model Comparison Plots
- Side-by-side performance comparison
- Grid plots for multiple models
- Confusion matrices

## Project Structure

```
dom-formula-assignment/
├── run_pipeline.py              # Main entry point
├── readme.md                    # This file
├── pipeline/                    # Main pipeline package
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── data_loader.py          # Data loading
│   ├── model_trainer.py        # Model training
│   ├── predictor.py            # Prediction logic
│   ├── evaluator.py            # Evaluation metrics
│   ├── pipeline_manager.py     # Pipeline orchestration
│   ├── plotting.py             # Visualization
│   ├── logger.py               # Logging utilities
│   └── utils/                  # Utility functions
│       ├── utils.py
│       └── generate_formula_combinations.py
├── data/                        # Data directory
│   ├── DOM_training_set_ver2/  # 7T training data
│   ├── DOM_training_set_ver3/  # 21T training data
│   ├── DOM_testing_set/        # Test files
│   ├── DOM_testing_set_Peaklists/  # Peak lists
│   └── synthetic_data/         # Synthetic formulas
├── models/                      # Trained models (auto-generated)
├── output/                      # Results (auto-generated)
│   ├── 7T/
│   ├── 21T/
│   ├── combined/
│   ├── synthetic_combined/
│   └── plots/
├── logs/                        # Execution logs (auto-generated)
├── tests/                       # Test scripts
└── dummy/                       # Development/testing files
```


### Logging

All operations are logged to `logs/pipeline_YYYYMMDD_HHMMSS.log`

Check logs for detailed error messages and execution traces.
