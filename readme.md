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



###  Run Specific Model Configuration

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

### Custom Training Data Configuration

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

###  Test on Specific Files

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
