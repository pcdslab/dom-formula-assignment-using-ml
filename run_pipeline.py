from pipeline.pipeline_manager import run_main
from pipeline.utils.generate_formula_combinations import generate_synthetic_data
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the machine learning pipeline.")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining of models")
    args = parser.parse_args()
    generate_synthetic_data()
    run_main(force_retrain=args.force_retrain)
