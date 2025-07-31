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
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text() for p in paragraphs)
    except Exception as e:
        print(f"Error fetching raw text for {source_title}: {e}")
        return ""
