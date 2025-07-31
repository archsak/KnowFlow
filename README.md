
# KnowFlow

KnowFlow is a modular system for identifying prerequisite concepts in text. It is designed to help readers understand complex documents by automatically extracting and ranking key expressions and terminology that are essential for comprehension.

## Project Overview
The system operates in three main stages:

1. **Potential Expression Identification (Stage A):**
   - Identifies candidate expressions in the text that could represent concepts. Wikipedia links are used as a means to help identify such concepts.
   - Uses a BERT-based classifier to detect relevant phrases.

2. **Content Domain Filtering (Stage B):**
   - Filters the candidate expressions to retain only those relevant to the main topic of the document.
   - Employs contextual embeddings and heuristics to assess domain relevance.

3. **Importance Ranking (Stage C):**
   - Ranks the filtered expressions on a scale from 0 to 3, according to their importance for understanding the text.
   - Uses a deep learning encoder model trained as a regressor with a penalty matrix and MSE loss; predictions are rounded to the nearest integer rank.

The system can be used both as a command-line pipeline and as a web service for uploading and analyzing documents.

## Repository Structure

- `main.py` — Orchestrates the full pipeline for all stages.
- `src/` — Source code for all stages and utilities.
  - `stage_a/LinkDetector.py` — Candidate expression identification (Stage A).
  - `stage_b/filter.py` — Content domain filtering (Stage B).
  - `stage_c/prerequisite_extractor_encoder.py` — Importance ranking (Stage C).
  - `util/` — Utilities for text extraction and preprocessing.
- `models/` — Pretrained model weights for each stage.
- `data/` — Contains raw and processed data used for training and evaluation.
- `uploads/`, `results/`, `static/`, `templates/` — Used by the web server for file uploads and result display.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/archsak/KnowFlow.git
   cd KnowFlow
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

3. **Download the required data:**
   - Pretrained model files should be placed in the `models/` directory.

## Running the Pipeline

You can run the full pipeline using the main script:

```bash
python main.py
```

This will execute all available stages in order, using the scripts and models present in the repository.

Alternatively, you can run each stage individually:

- **Stage A:**
  ```bash
  python src/stage_a/LinkDetector.py --mode train   # To train
  python src/stage_a/LinkDetector.py --mode predict # To predict links
  ```
- **Stage B:**
  ```bash
  python src/stage_b/filter.py
  ```
- **Stage C:**
  ```bash
  python src/stage_c/prerequisite_extractor_encoder.py
  ```

## Web Server

To use the web interface for uploading and analyzing documents:

```bash
python src/server.py
```
Then open [http://localhost:5002](http://localhost:5002) in your browser.

Note: if you run it with CPU it will take several minutes to get the results

## Reproducibility

To reproduce the results:
1. Ensure all required data files and pretrained models are in place.
2. Run the pipeline as described above.
3. Results will be saved in the `results/` directory and can be compared to reference outputs.

## System Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.5+
- NumPy, Pandas, scikit-learn, NLTK, spaCy, tqdm, PyMuPDF, python-docx, werkzeug, xgboost

## Authors
Miriam Pasikov  
Lipaz Holzman  
Aharon Sinai