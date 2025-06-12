import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote

article_titles = [
    
    # Tagged for milestone
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
    "Pathetic fallacy",
    
    # Will tag before final submission for better results
    "ResNet",
    "Transformer (deep learning architecture)",
    "Generative adversarial network",
    "Stochastic gradient descent",
    "DropConnect",
    "Neural tangent kernel",
    "Graph neural network", 
    "Autoassociative memory", 
    "CRISPR",
    "Sanger sequencing",
    "Structure of DNA",
    "Framingham Heart Study",
    "Stanford prison experiment",
    "RNA interference",
    "Human papillomavirus",
    "LRRK2",
    "Blue Zones",
    "Gut microbiome",
    "General relativity",
    "Gravitational wave",
    "Bell's theorem",
    "Cosmic microwave background",
    "Michelson–Morley experiment",
    "Kerr–Newman metric",
    "Hawking radiation",
    "Quasicrystal",
    "Neutrino oscillation",
    "Fast radio burst",
    "RSA (cryptosystem)",
    "Fast Fourier transform",
    "Simplex algorithm",
    "Fermat's Last Theorem",
    "Gödel's incompleteness theorems",
    "Computability theory",
    "Borsuk–Ulam theorem",
    "Szemerédi's theorem",
    "Green–Tao theorem",
    "Lorenz attractor",
    "Elliptic curve primality proving",
    "Langlands program",
    "Khovanov homology",
    "Catalan's conjecture",
    "Tragedy of the Commons",
    "Nudge theory",
    "The Structure of Scientific Revolutions",
    "Game theory",
    "Lake Wobegon effect",
    "Cultural dimensions theory",
    "Capital asset pricing model",
    "Collective memory",
    "Ecological modernization",
    "The Rite of Spring",
    "Twelve-tone technique",
    "Extended technique",
    "Spectral music",
    "Algorithmic composition",
    "The Interpretation of Dreams",
    "Electronic literature",
    "Stylometry",
    "On the Origin of Species",
    "Clash of Civilizations",
    "Annales school",
    "Medieval Iceland",
    "Anthropocene",
    "Citizens United v. FEC",
    "Regulation of artificial intelligence",
    "Open Government Partnership",
    "Human Genome Project",
    "Genome-wide association study",
    "Zebrafish",
    "Bootstrap aggregating",
    "False discovery rate",
    "Approximate entropy",
    "Topological data analysis",
    "Bitcoin",
    "Raft (computer science)",
    "Micropayment",
    "Peer review",
    "Open access",
    "Retraction",
    "Open Science"
    "Prospect theory",
    "Cumulative prospect theory",
    "Null hypothesis significance testing",
    "P-value",
    "Statistical significance",
    "Certainty effect",
    "Asch conformity experiments",
    "Milgram experiment",
    "Bystander effect",
    "Tuskegee Syphilis Study",
    "More Product, Less Process",
    "Silent Spring",
    "Bonferroni correction",
    "Type II error",
    "One-tailed test",
]



def extract_filtered_links(source_title):
    url = f"https://en.wikipedia.org/wiki/{source_title.replace(' ', '_')}"
    try:
        response = requests.get(url)
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
            
            
            # Check if this <a> is inside a table with a <td class="sidebar-pretitle"> containing "Part of a series on"
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



# go over all of the articles and extract the filetered links from them
def write_all_links_to_csv(article_titles, filename="check3.csv"):
    seen_pairs = set()  
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source article", "linked article", "concept", "score"])  

        for title in article_titles:
            print(f"Processing: {title}")
            links = extract_filtered_links(title)
            for source_title, linked_title, concept in links:
                pair = (source_title, concept.lower())
                if pair not in seen_pairs:
                    writer.writerow([source_title, concept, linked_title, ""])
                    seen_pairs.add(pair)


write_all_links_to_csv(article_titles)





# CODE FOR CHANGING : TO . IN NAMES OF FILES
# import pandas as pd

# # Path to your CSV
# csv_path = "data/raw/ranked_pages/rated_wiki_pages.csv"

# # Load, modify, and save back
# df = pd.read_csv(csv_path)

# if 'source_article' in df.columns:
#     df['source_article'] = df['source_article'].str.replace(":", ".", regex=False)
#     df.to_csv(csv_path, index=False)
#     print("✅ Updated source_article values and saved CSV.")
# else:
#     print("❌ 'source_article' column not found in the CSV.")
