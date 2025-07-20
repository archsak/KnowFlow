import subprocess
import sys

def run_script(script_path):
    """Executes a Python script and prints its output."""
    print("-" * 80)
    print(f"Executing: {script_path}")
    print("-" * 80)
    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        rc = process.poll()
        if rc == 0:
            print(f"\n--- Successfully finished {script_path} ---")
        else:
            print(f"\n--- Error running {script_path} (Exit Code: {rc}) ---")
        
        return rc
        
    except Exception as e:
        print(f"Failed to execute {script_path}. Error: {e}")
        return -1

def main():
    """Runs the training scripts for both models."""
    print("Starting the model training and evaluation pipeline.")
    
    # Run the feature-based model training
    run_script("src/stage_c/train_ranker_features.py")
    
    # Run the encoder-based model training
    run_script("src/stage_c/train_ranker_encoder_clean.py")
    
    print("\nAll training scripts have been executed.")

if __name__ == "__main__":
    main()