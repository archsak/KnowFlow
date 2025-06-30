# knowflow_pipeline.py

import os
import re
import json
import time
import torch
import random
import requests
import argparse
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import unquote
from typing import List, Dict
from datetime import datetime
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------
# Wikipedia data extraction
# ---------------------------------------

def extract_filtered_links(source_title):
    time.sleep(1.0)
    url = f"https://en.wikipedia.org/wiki/{source_title.replace(' ', '_')}"
    headers = {'User-Agent': 'KnowFlowBot/1.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except:
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    content_div = soup.find("div", {"id": "bodyContent"})
    if not content_div:
        return []

    links = []
    stop_headings = {"See_also", "Notes", "References", "Further_reading", "External_links"}
    for tag in content_div.descendants:
        if tag.name == "h2" and tag.get("id") in stop_headings:
            break
        if tag.name == "a" and tag.has_attr("href"):
            href = tag['href']
            if href.startswith("/wiki/") and ':' not in href and not href.startswith("#"):
                linked_title = unquote(href.split("/wiki/")[-1].replace('_', ' '))
                links.append((source_title, tag.get_text(), linked_title))
    return links

def get_clean_article_text(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "prop": "revisions", "rvslots": "main",
        "rvprop": "content", "format": "json", "titles": title
    }
    response = requests.get(url, params=params).json()
    pages = response['query']['pages']
    content = next(iter(pages.values()))['revisions'][0]['slots']['main']['*']
    return content.replace("\n", " ").replace("", " ")

def fetch_articles(titles: List[str]) -> List[Dict]:
    data = []
    for title in titles:
        print(f"Fetching: {title}")
        text = get_clean_article_text(title)
        links = extract_filtered_links(title)
        link_set = {display.lower() for _, display, _ in links}
        data.append({"title": title, "text": text, "links": list(link_set)})
    return data

# ---------------------------------------
# Dataset and model
# ---------------------------------------

class LinkDetectionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.samples = []
        for item in data:
            tokens = item['text'].split()
            candidates = [
                ' '.join(tokens[i:j]) for i in range(len(tokens))
                for j in range(i+1, min(i+5, len(tokens)+1))
            ]
            for phrase in candidates:
                label = int(phrase.lower() in item['links'])
                context = item['text']
                enc = tokenizer(context, phrase, truncation=True,
                                padding='max_length', max_length=max_len,
                                return_tensors='pt')
                self.samples.append((enc, label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        enc, label = self.samples[idx]
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTLinkClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask).pooler_output
        return self.out(self.drop(out))

# ---------------------------------------
# Training and evaluation
# ---------------------------------------

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            preds = outputs.argmax(dim=1)
            y_true += labels.cpu().tolist()
            y_pred += preds.cpu().tolist()
    print(classification_report(y_true, y_pred, digits=3))

# ---------------------------------------
# Entry point
# ---------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--articles", nargs='+', required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    articles = fetch_articles(args.articles)
    dataset = LinkDetectionDataset(articles, tokenizer)
    loader = DataLoader(dataset, batch_size=16)

    model = BERTLinkClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    if args.mode == "train":
        train_model(model, loader, optimizer, criterion, device)
        torch.save(model.state_dict(), "link_model.pt")
    else:
        model.load_state_dict(torch.load("link_model.pt", map_location=device))
        evaluate_model(model, loader, device)

if __name__ == "__main__":
    main()
