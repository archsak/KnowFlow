# KnowFlow
Prerequisite Concept Identification System for Texts
This project implements a machine learning model for identifying expressions and terminology that serve as prerequisites for understanding text. The system generates a list of concepts that should be familiar before reading the text.

# Project Goal
The goal is to help readers understand complex texts by identifying and classifying key concepts that are critical to understanding the content. The system is built in a modular way with three stages:

Potential Expression Identification: Identifying expressions that would constitute hyperlinks in a Wikipedia page
Content Domain Filtering: Analyzing the relevance of expressions to the content domain of the document
Importance Ranking: Ranking expressions on a scale of 0-3 according to their importance for understanding the text

#The Model
The system operates in three stages:

Stage A: Potential Expression Identification
Training on Wikipedia pages to identify expressions that constitute hyperlinks
Classification model that marks words or expressions as candidates to be links
Stage B: Content Domain Filtering
Creating vector representations (embeddings) for each expression and for the entire article
Calculating cosine similarity between the article vector and the expression vectors
Filtering expressions that pass a minimum similarity threshold indicating they belong to the same content domain
Stage C: Importance Ranking
Ranking expressions on a scale of 0-3 according to their importance for understanding the text, based on a supervised model trained on human-ranked data:
0: Not related to the content domain
1: Related to the content domain but not essential for understanding
2: Moderately important for understanding the text
3: Critical to understanding the text, an essential prerequisite

Installation and Execution
bash
# Clone the repository
git clone https://github.com/archsak/KnowFlow.git

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download data (if needed)
python src/utils/download_data.py

# Run model stages
python src/stage_a/train.py
python src/stage_b/filter.py

# Stage C has two different implementations that can be compared:
# Feature-based model (uses manually engineered features)
python src/stage_c/train_ranker.py
python src/stage_c/prerequisite_extractor.py

# Encoder-based model (uses deep learning encoder)
python src/stage_c/train_ranker_encoder.py
python src/stage_c/prerequisite_extractor_encoder.py

# Compare model results
python src/stage_c/compare_models.py

# System Requirements
Python 3.8+
PyTorch 1.9+
Transformers 4.5+
NumPy, Pandas, Scikit-learn
NLTK

# Authors
Miriam Pasikov
Lipaz Holzman
Aharon Sinai
