# DOM Formula Assignment

A machine learning pipeline for assigning molecular formulas to dissolved organic matter (DOM) mass spectrometry data.

##  Abstract

A machine learning approach to molecular formula assignment is crucial for unlocking the full potential of ultra-high resolution mass spectrometry (UHRMS) when analyzing complex mixtures. By combining data-driven models with rigorous benchmarking, the accuracy, consistency, and speed in identifying plausible molecular formulas from vast spectral datasets can be improved. Compared with traditional de novo methods that rely heavily on rule-based heuristics and manual parameter tuning, machine learning approaches can capture complex patterns in data and adapt more readily to diverse sample types. In this paper, we describe the application of a machine learning method using the k-nearest neighbors (KNN) algorithm, trained on curated chemical formula datasets from UHRMS analyses of dissolved organic matter (DOM), covering the saline river continuum and tropical wet/dry season variability. The influence of the mass accuracy (training set with 0.15-1ppm) was evaluated on a blind test set of DOMs of different geographical origins. A Decision Tree Regressor (DTR) and Random Forest Regressor (RFR) based on mass accuracy (<1ppm) was used. Results from our ML models exhibit 43% more formulas annotated than traditional methods (5796 vs 4047), Model-Synthetic achieved 99.9% assignment rate and annotated/assigned 2x more formulas (8,268 vs 4047). DTR and RFR achieved formula-level accuracies (FA) of 86.5% and 60.4%, respectively. Overall, results show an increase in formula assignment when compared with traditional methods. This ultimately enables more reliable characterization of complex natural and engineered systems, supporting advances in fields such as environmental science, metabolomics, and petroleomics. Furthermore, the novel data set produced for this study is made publicly available, establishing an initial benchmark for molecular formula assignment in UHRMS using machine learning. 


## Requirements

- Python 3.11
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, wordcloud, joblib

## Install Anaconda
[Step by Step Guide to Install Anaconda](https://docs.anaconda.com/anaconda/install/)

## Clone the Repository

```bash
# Clone the repository and navigate into it
git clone https://github.com/pcdslab/dom-formula-assignment-using-ml.git 
cd dom-formula-assignment-using-ml
``` 

## Install Dependencies
```bash
# Install Conda environment and dependencies 
conda create -n graphdom python=3.11
conda activate graphdom
pip install -r requirements.txt
```

### Usage

#### Run KNN models

```python
python run_pipeline.py
```
If outputs do not exits, it would create an output folder and save the results in it.

#### Run Decision Tree Regressor (DTR) and Random Forest Regressor (RFR) models
```python
python DT_and_RF.py
```
### Training
Run the following command to train the models with your data files in data/DOM_training_set_ver2 (7T), data/DOM_training_set_ver3 (21T), data/synthetic_data.
```python
python run_pipeline.py --force-retrain
```
To train DTR and RFR, please update train.txt and test.txt files either in the data folder or pass them as command-line arguments and run the following command:
```python
python DT_and_RF.py --train data/train.txt --test data/test.txt --retrain
```
## Data
The data is organized into the following folders and files: The L1 is the data from the 7T with mass resolution of 1 PPM, L2 is the data from the 9.4T instrument with mass resolution of 0.2-0.4 PPM, and L3 is the data from 21T with mass resolution of 0.15 PPM. The synthetic_data folder contains synthetically generated formulas for training.

* readme.txt
* train.txt (L1 with Mobility Features)
* test.txt (L2-v2 with Mobility Features)
* DOM_testing_set (L2, 9.4T)
* DOM_testing_set_Peaklists (L2-Peaklists, 9.4T)
* DOM_training_set_ver2 (L1, 7T)
* DOM_training_set_ver3 (L3, 21T)
* synthetic_data (Synthetic)


## Model Configurations

The pipeline includes four standard model configurations:

### 1. Model-L1
- **Training Data**: DOM_training_set_ver2 (7T data)
- **Description**: Model trained on 7T mass spectrometry data

### 2. Model-L3
- **Training Data**: DOM_training_set_ver3 (21T data)
- **Description**: Model trained on 21T mass spectrometry data

### 3. Model-L1-L3 (Ensemble)
- **Training Data**: Both ver2 and ver3
- **Description**: Ensemble model combining 7T and 21T data

### 4. Model-Synthetic (Ensemble)
- **Training Data**: Combined 7T, 21T, and synthetic data
- **Description**: Enhanced with synthetically generated formulas

Each model is trained with multiple configurations:
- **K values**: 1, 3 (number of nearest neighbors)
- **Distance metrics**: Euclidean (p=2), Manhattan (p=1)

This results in 16 model variants (4 models × 2 K values × 2 metrics).


## Output Files

**Per-Test-File Results** (`results_*.csv`):

**Peak List Predictions** (`peak_list/*.csv`):

## Citation
This paper is currently under review.

## License and Usage Terms
This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this model and its derivatives, which include models trained on outputs from the model or datasets created from the model, is prohibited and requires prior approval. If you are a commercial entity, please contact the corresponding author.

## Contact
For any additional questions or comments, contact Fahad Saeed (fsaeed@fiu.edu).
