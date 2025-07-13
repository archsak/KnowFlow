import argparse
import os
import time
import json
import random
import requests
from typing import List, Tuple, Dict
from urllib.parse import unquote

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd

from bs4 import BeautifulSoup

class LinkDetectionDataset(Dataset):
    """PyTorch Dataset for BERT-based link detection."""
    def __init__(self, data: List[Tuple[str, str, int]], tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, phrase, label = self.data[idx]
        encoded = self.tokenizer(
            phrase,
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


class PreTokenizedDataset(Dataset):
    """
    A PyTorch dataset that serves pre-tokenized (input_ids, attention_mask, label) tensors.
    """
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_masks[idx],
            self.labels[idx]
        )


class LinkClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


def batch_tokenize(phrases, texts, tokenizer, batch_size=1024):
    input_ids_list = []
    attention_masks_list = []

    print(f"📦 Tokenizing {len(phrases)} samples in batches...")

    for i in range(0, len(phrases), batch_size):
        batch_phrases = phrases[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]

        enc = tokenizer(
            batch_phrases,
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        input_ids_list.append(enc['input_ids'])
        attention_masks_list.append(enc['attention_mask'])
        print(f"finished batch {i+1}")

    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)
    return input_ids, attention_masks

def extract_filtered_links(source_title: str) -> List[Tuple[str, str, str]]:
    print(f"🌐 Extracting links for article: {source_title}")
    time.sleep(1.0)
    url = f"https://en.wikipedia.org/wiki/{source_title.replace(' ', '_')}"
    headers = {'User-Agent': 'KnowFlowBot/1.0 (https://example.com/contact)'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch {source_title}: {e}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    content_div = soup.find("div", {"id": "bodyContent"})
    if not content_div:
        print("⚠️ No content div found.")
        return []

    links = []
    stop_headings = {"See_also", "Notes", "References", "Further_reading", "External_links", "Citations", "Bibliography", "Sources"}

    for tag in content_div.descendants:
        if tag.name == "h2" and tag.get("id") in stop_headings:
            break

        if tag.name == "a" and tag.has_attr("href"):
            href = tag['href']
            if tag.find_parent("table"):
                pretitle_cell = tag.find_parent("table").find("td", class_="sidebar-pretitle")
                if pretitle_cell and "Part of a series on" in pretitle_cell.text:
                    continue

            if (href.startswith("/wiki/") and not any(href.startswith(p) for p in ["/wiki/Special:", "/wiki/Help:", "/wiki/File:", "/wiki/Category:", "/wiki/Template:"]) and ":" not in href.split("/wiki/")[-1] and not href.startswith("#")):
                linked_title = unquote(href.split("/wiki/")[-1].replace('_', ' '))
                links.append((source_title, tag.get_text(), linked_title))

    print(f"✅ Found {len(links)} valid links in {source_title}")
    return links


def get_raw_text(source_title: str) -> str:
    print(f"📝 Fetching raw text for: {source_title}")
    url = f"https://en.wikipedia.org/wiki/{source_title.replace(' ', '_')}"
    headers = {'User-Agent': 'KnowFlowBot/1.0 (https://example.com/contact)'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text() for p in paragraphs)
    except Exception as e:
        print(f"❌ Error fetching raw text for {source_title}: {e}")
        return ""


# def generate_candidates(text: str) -> List[str]:
#     print("🔍 Generating candidate phrases...")
#     words = text.split()
#     candidates = list(set([" ".join(words[i:j]) for i in range(len(words)) for j in range(i+1, min(i+4, len(words)+1))]))
#     print(f"🧠 Generated {len(candidates)} candidate phrases")
#     return candidates

import spacy
nlp = spacy.load("en_core_web_sm")

def generate_candidates(text: str) -> List[str]:
    doc = nlp(text)
    candidates = list(set(chunk.text.lower().strip() for chunk in doc.noun_chunks))
    print('number of candidates: ', len(candidates))
    return candidates


def create_training_data(titles: List[str]) -> List[Tuple[str, str, int]]:
    print(f"📚 Creating training data for {len(titles)} articles...")
    data = []
    for title in titles:
        true_links = extract_filtered_links(title)
        # true_phrases = {phrase for _, phrase, _ in true_links}
        true_phrases = {phrase.strip().lower() for _, phrase, _ in true_links}
        raw_text = get_raw_text(title)
        candidates = generate_candidates(raw_text)
        for phrase in candidates:
            # label = 1 if phrase in true_phrases else 0
            label = 1 if phrase.strip().lower() in true_phrases else 0
            data.append((raw_text, phrase, label))
    print(f"✅ Finished preparing dataset with {len(data)} samples")
    return data



def predict(args):
    print("started predict")
    title = args.text_file
    raw_text = get_raw_text(title)
    candidates = generate_candidates(raw_text)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = LinkClassifier().cuda()
    model.load_state_dict(torch.load("/content/drive/MyDrive/link_detector.pt"))
    model.eval()

    results = []
    for phrase in candidates:
        encoded = tokenizer(
            phrase,
            raw_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].cuda()
        attention_mask = encoded['attention_mask'].cuda()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[0]
            if probs[1].item() >= args.threshold:
                results.append(phrase)  # ⬅️ Only save the phrase

    results = sorted(set(results))  # optional: remove duplicates and sort alphabetically

    # Save to CSV
    safe_title = title.replace(" ", "_").replace("/", "_")
    output_dir = os.path.join("data", "processed", "stage_a")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{safe_title}.csv")

    df = pd.DataFrame(results, columns=["phrase"])  # ⬅️ Only one column: phrase
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(results)} predicted link phrases to {output_path}")



def train(args):
    print("📥 Loading training titles from training_titles.json...")
    with open("training_titles.json") as f:
        articles = json.load(f)

    print(f"🧠 Generating training data from {len(articles)} articles...")
    train_data = create_training_data(articles)
    print(f"✅ Created dataset with {len(train_data)} (text, phrase, label) samples.")

    print("🔤 Pre-tokenizing all input pairs...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    phrases = [phrase for _, phrase, _ in train_data]
    texts = [text for text, _, _ in train_data]
    labels = torch.tensor([label for _, _, label in train_data])

    print("hi")
    input_ids, attention_masks = batch_tokenize(phrases, texts, tokenizer)
    print("hi1")

    dataset = PreTokenizedDataset(
        input_ids=input_ids,
        attention_masks=attention_masks,
        labels=labels
    )
    print("hi2")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(f"🚀 Starting training with batch_size={args.batch_size}, epochs={args.epochs}")
    model = LinkClassifier().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).cuda())

    model.train()
    total_loss = 0
    final_preds, final_labels = [], []

    for epoch in range(args.epochs):
        print(f"\n🔁 Epoch {epoch+1}/{args.epochs}")
        epoch_start = time.time()
        all_preds = []
        all_labels = []

        for step, (input_ids, attention_mask, labels) in enumerate(loader, 1):
            input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
            all_preds.extend(batch_preds)
            all_labels.extend(labels.detach().cpu().tolist())

            if step % 100 == 0:
                print(f"  Step {step} — Loss: {loss.item():.4f}")

        print(f"✅ Finished Epoch {epoch+1} in {time.time() - epoch_start:.2f} seconds")

        if epoch == args.epochs - 1:
            final_preds = all_preds
            final_labels = all_labels

    print("\n💾 Saving model to saved_model/link_detector.pt")
    os.makedirs("saved_model", exist_ok=True)
    torch.save(model.state_dict(), "saved_model/link_detector.pt")

    precision, recall, f1, _ = precision_recall_fscore_support(final_labels, final_preds, average='binary')
    print("\n📊 Final Training Evaluation Metrics:")
    print(f"Total Examples: {len(final_labels)}")
    print(f"Positive Examples (links): {sum(final_labels)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average Loss: {total_loss / (args.epochs * len(loader)):.4f}")




def evaluate(args):
    with open("eval_titles.json") as f:
        articles = json.load(f)
    eval_data = create_training_data(articles)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = LinkDetectionDataset(eval_data, tokenizer)
    loader = DataLoader(dataset, batch_size=16)

    model = LinkClassifier().cuda()
    # model.load_state_dict(torch.load("saved_model/link_detector.pt"))
    # model.load_state_dict(torch.load("/content/drive/MyDrive/link_detector.pt"))
    model.load_state_dict(torch.load(args.model_path))


    model.eval()

    preds, labels = [], []
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in loader:
            input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
            batch_labels = batch_labels.cuda()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, batch_labels)
            total_loss += loss.item()
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)
            labels.extend(batch_labels.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    print("\n📊 Evaluation Metrics:")
    print(f"Total Examples: {len(labels)}")
    print(f"Positive Examples (links): {sum(labels)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average Loss: {total_loss / len(loader):.4f}")
    print("\nDetailed Report:")
    print(classification_report(labels, preds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'], default='train')
    parser.add_argument('--articles', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--text_file', type=str)
    parser.add_argument('--model_path', type=str, default='saved_model/link_detector.pt')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
    elif args.mode == 'evaluate':
        evaluate(args)
        
        
if __name__ == '__main__':
    print("hereeee")
    main()