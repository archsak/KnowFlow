import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from tqdm import tqdm
import sys
import random

# Add src to path for imports
sys.path.append('src')
sys.path.append('src/stage_a')
from src.util.get_raw_text import get_raw_text

# Load spaCy for concept normalization

# ...existing code from train_ranker_encoder_clean.py...
