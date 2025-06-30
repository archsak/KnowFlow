"""
Wikipedia Data Extractor for KnowFlow Project (MODIFIED)
Fetches real Wikipedia articles with actual link markup for training
"""

import requests
import json
import re
from typing import List, Dict, Tuple, Optional
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import unquote
import mwparserfromhell

class WikipediaExtractor:
    """Extracts Wikipedia articles with real link information."""

    def get_article_data(self, title: str) -> Optional[Dict]:
        """
        Get Wikipedia article with clean text and link information.

        Returns:
            Dict with 'title', 'clean_text', and 'links'
        """
        try:
            raw_links = extract_filtered_links(title)

            clean_text = get_clean_article_text_until_stop(title)

            seen_pairs = set()
            formatted_links = []
            for source_title, display_text, linked_title in raw_links:
                pair = (source_title, display_text.lower())
                if pair not in seen_pairs:
                    formatted_links.append({
                        'target': linked_title,
                        'display_text': display_text,
                        'start_pos': None,
                        'end_pos': None
                    })
                    seen_pairs.add(pair)

            return {
                'title': title,
                'clean_text': clean_text,
                'raw_markup': None,
                'links': formatted_links
            }

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return None


    def get_training_dataset(self, article_titles: List[str], save_cache: bool = True) -> List[Dict]:
        """
        Fetch multiple Wikipedia articles for training.
        """
        training_data = []
        successful_fetches = 0

        print(f"Fetching {len(article_titles)} Wikipedia articles...")

        for i, title in enumerate(article_titles):
            print(f"Fetching {i+1}/{len(article_titles)}: {title}")
            article_data = self.get_article_data(title)

            if article_data and len(article_data['links']) > 0:
                training_data.append(article_data)
                successful_fetches += 1
                print(f"Got {len(article_data['links'])} links")
            else:
                print(f"Failed or no links found")

            time.sleep(0.1)

        print(f"\nSuccessfully fetched {successful_fetches}/{len(article_titles)} articles")

        if save_cache and training_data:
            self.save_dataset(training_data, 'wikipedia_cache.json')

        return training_data

    def save_dataset(self, dataset: List[Dict], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to {filename}")

    def load_dataset(self, filename: str = 'wikipedia_cache.json') -> List[Dict]:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            print(f"Dataset loaded from {filename} ({len(dataset)} articles)")
            return dataset
        except FileNotFoundError:
            print(f"Cache file {filename} not found")
            return []



def extract_filtered_links(source_title):
    time.sleep(1.0)  # Respect Wikipedia's rate limits

    url = f"https://en.wikipedia.org/wiki/{source_title.replace(' ', '_')}"
    headers = {
        'User-Agent': 'KnowFlowBot/1.0 (https://example.com/contact)'  # <-- Customize this
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {source_title}: {e}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    content_div = soup.find("div", {"id": "bodyContent"})
    if not content_div:
        return []

    links = []
    stop_headings = {
        "See_also", "Notes", "References", "Further_reading", "External_links",
        "Citations", "Bibliography", "Sources"
    }

    for tag in content_div.descendants:
        if tag.name == "h2":
            if tag.get("id") in stop_headings:
                break

        if tag.name == "a" and tag.has_attr("href"):
            href = tag['href']

            parent_table = tag.find_parent("table")
            if parent_table:
                pretitle_cell = parent_table.find("td", class_="sidebar-pretitle")
                if pretitle_cell and "Part of a series on" in pretitle_cell.text:
                    continue

            if (
                href.startswith("/wiki/")
                and not any(href.startswith(prefix) for prefix in [
                    "/wiki/Special:", "/wiki/Help:", "/wiki/File:", "/wiki/Category:", "/wiki/Template:"
                ])
                and ":" not in href.split("/wiki/")[-1]
                and not href.startswith("#")
            ):
                linked_title = unquote(href.split("/wiki/")[-1].replace('_', ' '))
                links.append((source_title, tag.get_text(), linked_title))

    return links

def get_clean_article_text_until_stop(title):
    stop_headings = {
        "See also", "Notes", "References", "Further reading", "External links",
        "Citations", "Bibliography", "Sources"
    }

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "rvslots": "main",
        "rvprop": "content",
        "format": "json",
        "titles": title
    }

    response = requests.get(url, params=params)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    wikitext = page["revisions"][0]["slots"]["main"]["*"]

    wikicode = mwparserfromhell.parse(wikitext)
    lines = []
    reached_stop = False

    for section in wikicode.get_sections(include_lead=True, levels=[2]):
        heading = section.filter_headings()
        if heading:
            heading_text = heading[0].title.strip_code().strip()
            if heading_text in stop_headings:
                reached_stop = True
                break
        if not reached_stop:
            lines.append(section.strip_code().strip())

    return "\n\n".join(lines)


def get_diverse_article_titles() -> List[str]:
    return [
        "Prime Number",
        "Attention Is All You Need",
        "BERT (language model)",
        "GPT (language model)",
        "AlexNet",
        "Word2vec",
        "U-Net",
        "Capsule neural network",
        "Neural differential equation",
        "DeepDream",
        "Batch normalization",
        "Swish function",
        "Microhistory",
        "Eurasia Group",
        "Miranda v. Arizona",
        "Cook–Levin theorem",
        "Beowulf: The Monsters and the Critics",
        "Pathetic fallacy"
    ]

def get_random_articles(num_articles: int = 10) -> List[str]:
    session = requests.Session()
    session.headers.update({'User-Agent': 'KnowFlow/1.0 (Educational Research)'})

    titles = []
    while len(titles) < num_articles:
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'random',
                'rnnamespace': 0,
                'rnlimit': min(50, num_articles - len(titles))  # max 50 per request
            }
            response = session.get("https://en.wikipedia.org/w/api.php", params=params)
            response.raise_for_status()
            data = response.json()
            titles += [page['title'] for page in data['query']['random']]
        except Exception as e:
            print(f"Error during API call: {e}")
            break

    return list(set(titles))[:num_articles]  # deduplicate just in case


# if __name__ == "__main__":
#     extractor = WikipediaExtractor()
#     articles = get_diverse_article_titles()
#     articles = extractor.get_training_dataset(
#         ["Prime Number"],
#         save_cache=False
#     )

    
 