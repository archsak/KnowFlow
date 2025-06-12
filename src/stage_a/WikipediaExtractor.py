"""
Wikipedia Data Extractor for KnowFlow Project
Fetches real Wikipedia articles with actual link markup for training
"""

import requests
import json
import re
from typing import List, Dict, Tuple, Optional
import time
import random
from bs4 import BeautifulSoup


class WikipediaExtractor:
    """Extracts Wikipedia articles with real link information."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'KnowFlow/1.0 (Educational Research)'
        })
        self.base_url = "https://en.wikipedia.org/w/api.php"
    
    def get_article_data(self, title: str) -> Optional[Dict]:
        """
        Get Wikipedia article with clean text and link information.
        
        Returns:
            Dict with 'title', 'clean_text', 'raw_markup', and 'links'
        """
        try:
            # Get raw wikitext with links
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'revisions',
                'rvprop': 'content',
                'rvslots': 'main'
            }
            
            response = self.session.get(self.base_url, params=params)
            data = response.json()
            
            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            
            if page_id == '-1':
                print(f"Page '{title}' not found")
                return None
            
            raw_markup = pages[page_id]['revisions'][0]['slots']['main']['*']
            clean_text, links = self._process_markup(raw_markup)
            
            return {
                'title': title,
                'clean_text': clean_text,
                'raw_markup': raw_markup,
                'links': links
            }
            
        except Exception as e:
            print(f"Error fetching '{title}': {e}")
            return None
    
    def _process_markup(self, markup: str) -> Tuple[str, List[Dict]]:
        """Process Wikipedia markup to extract clean text and links."""
        
        # Remove unwanted sections before processing
        markup = self._remove_metadata_sections(markup)
        markup = self._clean_markup(markup)
        
        links = []
        clean_text = markup
        
        # Extract internal links: [[Target|Display]] or [[Target]]
        link_pattern = r'\[\[([^\]|#\n]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]'
        
        for match in re.finditer(link_pattern, markup):
            target = match.group(1).strip()
            display_text = match.group(2).strip() if match.group(2) else target
            
            if self._should_include_link(target, display_text):
                links.append({
                    'target': target,
                    'display_text': display_text,
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        # Create clean text by replacing link markup with display text
        clean_text = re.sub(link_pattern, 
                           lambda m: m.group(2) if m.group(2) else m.group(1), 
                           clean_text)
        
        # Final cleanup
        clean_text = self._final_text_cleanup(clean_text)
        
        return clean_text, links
    
    def _remove_metadata_sections(self, markup: str) -> str:
        """Remove references, external links, and other metadata sections."""
        # Split at common metadata sections
        section_patterns = [
            r'(?i)==\s*(references?|sources?|further reading|external links|see also|notes|explanatory notes)\s*==',
        ]
        
        for pattern in section_patterns:
            sections = re.split(pattern, markup, maxsplit=1)
            if len(sections) > 1:
                markup = sections[0]  # Keep only content before metadata
                break
        
        return markup
    
    def _clean_markup(self, markup: str) -> str:
        """Clean Wikipedia markup and remove unwanted elements including series tables."""
        # 1. Remove references, templates, comments, etc. (regex-based cleanup)
        markup = re.sub(r'<ref[^>]*>.*?</ref>', '', markup, flags=re.DOTALL)
        markup = re.sub(r'<ref[^>]*/?>', '', markup)
        markup = re.sub(r'\{\{[^}]*\}\}', '', markup)
        markup = re.sub(r'\{\|[^}]*\|\}', '', markup, flags=re.DOTALL)
        markup = re.sub(r'\[\[Category:[^\]]*\]\]', '', markup)
        markup = re.sub(r'\[\[File:[^\]]*\]\]', '', markup)
        markup = re.sub(r'\[\[Image:[^\]]*\]\]', '', markup)
        markup = re.sub(r'<!--.*?-->', '', markup, flags=re.DOTALL)

        # 2. Use BeautifulSoup to remove unwanted links in sidebar tables
        soup = BeautifulSoup(markup, 'html.parser')
        
        # Find all <a> tags
        for tag in soup.find_all('a'):
            parent_table = tag.find_parent('table')
            if parent_table:
                pretitle_cell = parent_table.find('td', class_='sidebar-pretitle')
                if pretitle_cell and 'Part of a series on' in pretitle_cell.text:
                    tag.decompose()  # remove the <a> tag completely
        
        return str(soup)

    
    def _should_include_link(self, target: str, display_text: str) -> bool:
        """Determine if a link should be included in training data."""
        
        # Skip namespace links
        skip_prefixes = [
            'File:', 'Image:', 'Category:', 'Template:', 'Help:',
            'Portal:', 'Wikipedia:', 'WP:', 'User:', 'Talk:', 'Special:',
            'Media:', 'Commons:'
        ]
        
        for prefix in skip_prefixes:
            if target.startswith(prefix):
                return False
        
        # Skip overly short or numeric-only targets
        if len(target) < 3 or target.isdigit(): #todo should i delete < 3?
            print('target: ', target) #todo delete
            return False
        
        # Skip year-only links (often over-linked)
        if re.match(r'^\d{4}$', target):
            return False
        
        # Skip if display text is too short
        if len(display_text) < 2:
            return False
        
        return True
    
    def _final_text_cleanup(self, text: str) -> str:
        """Final cleanup of extracted text."""
        # Remove remaining markup
        text = re.sub(r"'{2,}", '', text)  # Bold/italic markup
        text = re.sub(r'<[^>]+>', '', text)  # HTML tags
        text = re.sub(r'\{\{[^}]*\}\}', '', text)  # Remaining templates
        
        # Clean whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text
    
    def get_training_dataset(self, article_titles: List[str], 
                           save_cache: bool = True) -> List[Dict]:
        """
        Fetch multiple Wikipedia articles for training.
        
        Args:
            article_titles: List of Wikipedia article titles
            save_cache: Whether to save fetched data to file
            
        Returns:
            List of article data dictionaries
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
            
            # Be respectful to Wikipedia servers
            time.sleep(0.1)
        
        print(f"\nSuccessfully fetched {successful_fetches}/{len(article_titles)} articles")
        
        if save_cache and training_data:
            self.save_dataset(training_data, 'wikipedia_cache.json')
        
        return training_data
    
    def save_dataset(self, dataset: List[Dict], filename: str):
        """Save dataset to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to {filename}")
    
    def load_dataset(self, filename: str = 'wikipedia_cache.json') -> List[Dict]:
        """Load dataset from JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            print(f"Dataset loaded from {filename} ({len(dataset)} articles)")
            return dataset
        except FileNotFoundError:
            print(f"Cache file {filename} not found")
            return []


def get_diverse_article_titles() -> List[str]:
    """Get a curated list of diverse Wikipedia articles for training."""
    return [
        # Computer Science & Technology
        "Machine learning", "Artificial intelligence", "Deep learning",
        "Neural network", "Python (programming language)", "Algorithm",
        "Computer science", "Data science", "Natural language processing",
        "Software engineering", "Database", "Operating system",
        
        # Science & Mathematics  
        "Physics", "Mathematics", "Biology", "Chemistry", "Statistics",
        "Quantum mechanics", "Theory of relativity", "Evolution",
        "Genetics", "Calculus", "Linear algebra",
        
        # History & Geography
        "World War II", "Renaissance", "Ancient Rome", "United States",
        "Europe", "Asia", "Climate change", "French Revolution",
        "Industrial Revolution", "Cold War",
        
        # Arts & Culture
        "Literature", "Music", "Painting", "Philosophy", "Psychology",
        "Shakespeare", "Classical music", "Modern art",
        
        # Medicine & Health
        "Medicine", "Human body", "Neuroscience", "Immunology",
        "Cardiovascular system", "Mental health",
        
        # Economics & Politics
        "Economics", "Democracy", "Capitalism", "International relations",
        "Macroeconomics", "Political science"
    ]


def get_random_articles(num_articles: int = 10) -> List[str]:
    """Get random Wikipedia article titles using the API."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'KnowFlow/1.0 (Educational Research)'
    })
    
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,
            'rnlimit': num_articles
        }
        
        response = session.get("https://en.wikipedia.org/w/api.php", params=params)
        data = response.json()
        
        return [page['title'] for page in data['query']['random']]
        
    except Exception as e:
        print(f"Error getting random articles: {e}")
        return get_diverse_article_titles()[:num_articles]


if __name__ == "__main__":
    # Demo usage
    extractor = WikipediaExtractor()
    
    # Test single article
    article_data = extractor.get_article_data("Artificial intelligence")
    if article_data:
        print(f"Title: {article_data['title']}")
        print(f"Links found: {len(article_data['links'])}")
        print(f"Links found, actual: {(article_data['links'])}")
        for link in article_data['links']:
            print(link['display_text'])
        # print(f"Text: {(article_data['clean_text'])}")
        print(f"Text length: {len(article_data['clean_text'])}")
        
        # Show sample links
        for link in article_data['links'][:5]:
            print(f"  Link: '{link['display_text']}' -> {link['target']}")