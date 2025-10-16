"""
DOM Formula Assignment Pipeline Package

This package provides a modular pipeline for DOM (Dissolved Organic Matter) 
formula assignment using K-Nearest Neighbors (KNN) machine learning.

Components:
- data_loader: Handles loading and preprocessing of training and testing data
- model_trainer: Manages KNN model training and persistence  
- predictor: Makes predictions on peak lists and test data
- evaluator: Calculates evaluation metrics and generates reports
- config: Centralized configuration management
- pipeline_manager: Orchestrates the entire pipeline workflow
- plotting: Visualization and plotting functions
- logger: Logging utilities
- utils: Utility functions

Main entry points:
- PipelineManager: Main class for running pipelines
- run_main(): Function to run all standard pipelines
"""

from .pipeline_manager import PipelineManager, run_main, run_pipeline_from_folder
from .data_loader import DataLoader
from .model_trainer import ModelTrainer
from .predictor import Predictor
from .evaluator import Evaluator
from .config import ConfigManager, PipelineConfig, DataConfig, ModelConfig

__version__ = "1.0.0"
__author__ = "DOM Formula Assignment Team"

__all__ = [
    "PipelineManager",
    "DataLoader", 
    "ModelTrainer",
    "Predictor",
    "Evaluator",
    "ConfigManager",
    "PipelineConfig",
    "DataConfig", 
    "ModelConfig",
    "run_main",
    "run_pipeline_from_folder"
]