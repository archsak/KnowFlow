import requests
import time
from bs4 import BeautifulSoup
from typing import List

def get_raw_text(source_title: str) -> str:
    print(f"Fetching raw text for: {source_title}")
    url = f"https://en.wikipedia.org/wiki/{source_title.replace(' ', '_')}"
    headers = {'User-Agent': 'KnowFlowBot/1.0 (https://example.com/contact)'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching page: {e}")
        return ""
    soup = BeautifulSoup(response.content, "html.parser")
    content_div = soup.find("div", {"id": "bodyContent"})
    if not content_div:
        print("No content found on page.")
        return ""
    # Extract all text from the content div
    text = content_div.get_text(separator=" ", strip=True)
    return text
