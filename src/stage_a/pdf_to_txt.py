import fitz as PyMuPDF  # Use a specific alias
import re

def pdf_to_clean_text(pdf_path: str, output_txt_path: str = None) -> str:
    """
    Convert a PDF to clean text and optionally save it to a .txt file.

    Args:
        pdf_path: Path to the input PDF.
        output_txt_path: Optional path to save the text file.

    Returns:
        Extracted plain text from the PDF.
    """
    doc = PyMuPDF.open(pdf_path)  # Use the alias here
    full_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        text = clean_page_text(text)
        full_text.append(text)

    final_text = '\n'.join(full_text).strip()

    # Save to text file if requested
    if output_txt_path:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        print(f"Saved extracted text to: {output_txt_path}")

    return final_text

def clean_page_text(text: str) -> str:
    """Clean artifacts from a single page of PDF text."""
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = text.strip().split('\n')
    lines = [line for line in lines if not re.match(r'^(arXiv|Page|doi|Proceedings|Copyright|.*[0-9]{4})', line)]
    return '\n'.join(lines).strip()

if __name__ == "__main__":
    pdf_path = "example.pdf"
    txt_path = "example_arxiv_paper.txt"

    text = pdf_to_clean_text(pdf_path, output_txt_path=txt_path)

    # later:
    # from Bert import AdvancedLinkDetector
    # detector = AdvancedLinkDetector()
    # detector.predict_links(text)
