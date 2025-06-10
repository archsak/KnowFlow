import os
import sys
import subprocess

def run_stage(description, command):
    print(f"\n=== {description} ===")
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error running: {command}")
        sys.exit(1)

def main():
    if os.path.exists("src/stage_a/train.py"):
        run_stage("Stage A: Potential Expression Identification", "python src/stage_a/train.py")
    else:
        print("Stage A script not found, skipping.")

    if os.path.exists("src/stage_b/filter.py"):
        run_stage("Stage B: Content Domain Filtering", "python src/stage_b/filter.py")
    else:
        print("Stage B script not found, skipping.")

    if os.path.exists("src/stage_c/train_ranker_features.py"):
        run_stage("Stage C: Feature-based Model Training", "python src/stage_c/train_ranker_features.py")
    if os.path.exists("src/stage_c/prerequisite_extractor_features.py"):
        run_stage("Stage C: Feature-based Prediction", "python src/stage_c/prerequisite_extractor_features.py")
    if os.path.exists("src/stage_c/train_ranker_encoder.py"):
        run_stage("Stage C: Encoder-based Model Training", "python src/stage_c/train_ranker_encoder.py")
    if os.path.exists("src/stage_c/prerequisite_extractor_encoder.py"):
        run_stage("Stage C: Encoder-based Prediction", "python src/stage_c/prerequisite_extractor_encoder.py")
    if os.path.exists("src/stage_c/train_ranker_ensemble.py"):
        run_stage("Stage C: Ensemble Model Training", "python src/stage_c/train_ranker_ensemble.py")
    if os.path.exists("src/stage_c/prerequisite_extractor_ensemble.py"):
        run_stage("Stage C: Ensemble Prediction", "python src/stage_c/prerequisite_extractor_ensemble.py")

    if os.path.exists("src/stage_c/compare_models.py"):
        run_stage("Compare Models", "python src/stage_c/compare_models.py")

    print("\nAll stages completed successfully.")

if __name__ == "__main__":
    main()
