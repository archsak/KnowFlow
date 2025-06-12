import requests
import mwparserfromhell
import os

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
    "Open Science",
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

# Example usage
output_dir = "data/raw/raw_texts"

for title in article_titles:
    safe_title = title.replace(" ", "_").replace("/", "_")
    filename = os.path.join(output_dir, f"{safe_title}.txt")
    
    try:
        text = get_clean_article_text_until_stop(title)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Saved: {filename}")
        
    except Exception as e:
        print(f"❌ Failed to save {filename}: {e}")
