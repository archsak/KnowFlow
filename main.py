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
    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "predict"]:
        print("Usage: python main.py [train|predict]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "train":
        if os.path.exists("src/stage_a/LinkDetector.py"):
            run_stage("Stage A: Potential Expression Identification", "python src/stage_a/LinkDetector.py")
        else:
            print("Stage A script not found, skipping.")

        if os.path.exists("src/stage_b/filter.py"):
            run_stage("Stage B: Content Domain Filtering", "python src/stage_b/filter.py")
        else:
            print("Stage B script not found, skipping.")

        if os.path.exists("src/stage_c/train_ranker_encoder.py"):
            run_stage("Stage C: Encoder-based Model Training", "python src/stage_c/train_ranker_encoder.py")
        else:
            print("Stage C training script not found, skipping.")

        print("\nAll training stages completed successfully.")

    elif mode == "predict":
        if os.path.exists("src/stage_c/prerequisite_extractor_encoder.py"):
            run_stage("Stage C: Encoder-based Prediction", "python src/stage_c/prerequisite_extractor_encoder.py")
        else:
            print("Stage C prediction script not found, skipping.")

        print("\nAll prediction stages completed successfully.")

if __name__ == "__main__":
    main()
