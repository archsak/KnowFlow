import os

def extract_article_text(title_or_path, search_dir=None):
    """
    Extract text from an article file (txt, pdf, docx) by path or by searching in a directory by title (without extension).
    If search_dir is None, treat title_or_path as a file path. If search_dir is given, search for file with supported extensions.
    """
    from src.util.pdf_to_txt import pdf_to_clean_text
    try:
        import docx
        def docx_to_text(path):
            doc = docx.Document(path)
            return "\n".join([para.text for para in doc.paragraphs])
        DOCX_SUPPORTED = True
    except ImportError:
        DOCX_SUPPORTED = False
    if search_dir is None:
        # Direct file path
        path = title_or_path
        ext = os.path.splitext(path)[1].lower()
        if not os.path.exists(path):
            return ''
        try:
            if ext == '.txt':
                with open(path, encoding='utf-8') as f:
                    return f.read()
            elif ext == '.pdf':
                return pdf_to_clean_text(path)
            elif ext == '.docx' and DOCX_SUPPORTED:
                return docx_to_text(path)
            else:
                return ''
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return ''
    else:
        # Search for file by title in directory
        for ext in ['.txt', '.pdf', '.docx']:
            candidate = os.path.join(search_dir, f'{title_or_path}{ext}')
            if os.path.exists(candidate):
                try:
                    if ext == '.txt':
                        with open(candidate, encoding='utf-8') as f:
                            return f.read()
                    elif ext == '.pdf':
                        return pdf_to_clean_text(candidate)
                    elif ext == '.docx' and DOCX_SUPPORTED:
                        return docx_to_text(candidate)
                    else:
                        return ''
                except Exception as e:
                    print(f"Error reading {candidate}: {e}")
                    return ''
        return ''
