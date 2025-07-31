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

    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "predict", "eval"]:
        print("Usage: python main.py [train|predict|eval]")
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

    elif mode == "eval":
        # Evaluation pipeline for articles with ranks 2 and 3 in eval_rated_pages.csv
        import pandas as pd
        import numpy as np
        import torch
        from transformers import AutoTokenizer
        sys.path.append('src')
        sys.path.append('src/util')
        from src.stage_c.train_ranker_encoder import PrerequisiteRankerModel, detailed_prediction_analysis, SoftPenaltyMatrixLoss
        from src.util.get_raw_text import get_raw_text

        MODEL_PATH = "models/stage_c_ranker_encoder_penalty.pt"
        MODEL_NAME = "bert-base-uncased"
        EVAL_CSV = "data/raw/ranked_pages/eval_rated_pages.csv"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(MODEL_PATH):
            print(f"Model not found at {MODEL_PATH}. Please train the model first.")
            sys.exit(1)
        if not os.path.exists(EVAL_CSV):
            print(f"Evaluation data not found at {EVAL_CSV}.")
            sys.exit(1)

        print(f"\n=== Evaluating model on {EVAL_CSV} ===")
        df = pd.read_csv(EVAL_CSV)
        # Standardize column names if needed
        df = df.rename(columns={
            'source_article': 'page_title',
            'concept': 'expression',
            'score': 'human_rank'
        })
        # Filter for ranks 2 and 3
        df = df[df['human_rank'].isin([2, 3])].reset_index(drop=True)
        if df.empty:
            print("No samples with human_rank 2 or 3 in evaluation data.")
            sys.exit(1)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = PrerequisiteRankerModel(MODEL_NAME).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        # Prepare article texts
        print("Caching article texts for evaluation...")
        article_texts = {title: get_raw_text(title) for title in df['page_title'].unique()}


        # Build a set of (page_title, expression) for ground truth and for predictions
        gt_dict = {}  # (page_title, expression) -> label
        for i, row in df.iterrows():
            gt_dict[(str(row['page_title']), str(row['expression']))] = int(row['human_rank'])

        pred_dict = {}  # (page_title, expression) -> predicted_rank (rounded)
        pred_raw_dict = {}  # (page_title, expression) -> predicted_rank (float)
        predictions = []
        labels = []
        with torch.no_grad():
            for i, row in df.iterrows():
                expr = str(row['expression'])
                page_title = str(row['page_title'])
                label = row['human_rank']
                context = article_texts.get(page_title, "")
                doc_snippet = context[:1000] if len(context) > 1000 else context
                input_text = f"Rate prerequisite importance (0-3): Is '{expr}' essential before reading '{page_title}'? Context: {doc_snippet}"
                encoding = tokenizer(
                    input_text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                output = model(input_ids, attention_mask)
                pred = output.item()
                pred_rounded = int(np.clip(round(pred), 0, 3))
                predictions.append(pred)
                labels.append(label)
                pred_dict[(page_title, expr)] = pred_rounded
                pred_raw_dict[(page_title, expr)] = pred

        predictions = np.array(predictions)
        labels = np.array(labels)

        # Error analysis: collect all types of errors
        error_rows = []
        gt_keys = set(gt_dict.keys())
        pred_keys = set(pred_dict.keys())

        # 1. Expressions in ground truth but not predicted (missing predictions)
        for key in gt_keys - pred_keys:
            page_title, expr = key
            error_rows.append({
                'page_title': page_title,
                'expression': expr,
                'error_type': 'missing_prediction',
                'true_rank': gt_dict[key],
                'predicted_rank': '',
                'predicted_rank_raw': ''
            })

        # 2. Expressions predicted but not in ground truth (extra predictions)
        for key in pred_keys - gt_keys:
            page_title, expr = key
            error_rows.append({
                'page_title': page_title,
                'expression': expr,
                'error_type': 'extra_prediction',
                'true_rank': '',
                'predicted_rank': pred_dict[key],
                'predicted_rank_raw': pred_raw_dict[key]
            })

        # 3. Expressions in both, but with different ranks (including 0/1)
        for key in gt_keys & pred_keys:
            page_title, expr = key
            true_rank = gt_dict[key]
            pred_rank = pred_dict[key]
            pred_rank_raw = pred_raw_dict[key]
            if true_rank != pred_rank:
                error_rows.append({
                    'page_title': page_title,
                    'expression': expr,
                    'error_type': 'wrong_rank',
                    'true_rank': true_rank,
                    'predicted_rank': pred_rank,
                    'predicted_rank_raw': pred_rank_raw
                })

        # Save all errors to CSV
        error_df = pd.DataFrame(error_rows)
        error_csv_path = 'results/eval_errors.csv'
        os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)
        error_df.to_csv(error_csv_path, index=False)
        print(f"\nSaved detailed error analysis to {error_csv_path}")

        # Use the same penalty matrix as in training
        penalty_matrix = np.array([
            [0.0, 3.0, 25.0, 100.0],
            [5.0, 0.0, 15.0, 80.0],
            [15.0, 8.0, 0.0, 12.0],
            [100.0, 60.0, 20.0, 0.0]
        ], dtype=np.float32)

        print("\nEvaluation Results:")
        detailed_prediction_analysis(predictions, labels, "Stage C Encoder Model (Eval)", penalty_matrix)
        print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
