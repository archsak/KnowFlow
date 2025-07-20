import os
import sys
import json
import http.server
import socketserver
import cgi
from werkzeug.utils import secure_filename

# --- Add src to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stage_a.pdf_to_txt import pdf_to_clean_text
from stage_a.Bert1 import KnowFlowBERT1Detector
from stage_b.filter import ContentDomainFilter
from stage_c.prerequisite_extractor_features import PrerequisiteRanker
from transformers import AutoTokenizer, AutoModel
import torch

# --- DOCX Support ---
try:
    import docx
    DOCX_SUPPORTED = True
    def docx_to_text(path):
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
except ImportError:
    DOCX_SUPPORTED = False

# --- Configuration ---
PORT = 5002
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Loading ---
print("Loading pipeline models...")

# Shared BERT model
print("Loading shared BERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device)


# Stage A
print("Loading Stage A: BERT1 Link Detector...")
# Make sure to use a valid model path for Bert1
BERT1_MODEL_PATH = 'models/bert1_link_detector.pt'
bert1_detector = KnowFlowBERT1Detector(
    model_path=BERT1_MODEL_PATH,
    bert_model=bert_model,
    tokenizer=tokenizer,
    device=device
)

# Stage B
print("Loading Stage B: Content Domain Filter...")
content_filter = ContentDomainFilter(
    bert_model=bert_model,
    tokenizer=tokenizer
)

# Stage C
print("Loading Stage C: Prerequisite Ranker (Features)...")
RANKER_MODEL_PATH = 'models/stage_c_ranker.joblib'
prerequisite_ranker = PrerequisiteRanker(model_path=RANKER_MODEL_PATH)

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/templates/index.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == '/upload':
            content_type, _ = cgi.parse_header(self.headers['Content-Type'])
            
            if content_type == 'multipart/form-data':
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST'})
                
                if 'file' in form:
                    file_item = form['file']
                    if file_item.filename:
                        filename = secure_filename(file_item.filename)
                        filepath = os.path.join(UPLOAD_FOLDER, filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(file_item.file.read())
                        
                        try:
                            response_data = self.process_file(filepath, filename)
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps(response_data).encode('utf-8'))
                        except Exception as e:
                            self.send_response(500)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            error_response = {'error': f'An error occurred: {str(e)}'}
                            self.wfile.write(json.dumps(error_response).encode('utf-8'))
                        finally:
                            if os.path.exists(filepath):
                                os.remove(filepath)
                        return

            self.send_error(400, "Bad Request")
            return

    def process_file(self, filepath, filename):
        text = ""
        if filename.lower().endswith('.pdf'):
            text = pdf_to_clean_text(filepath)
        elif DOCX_SUPPORTED and filename.lower().endswith('.docx'):
            text = docx_to_text(filepath)
        elif filename.lower().endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise Exception('Unsupported file type')

        if not text.strip():
            raise Exception('Could not extract text from file.')

        # --- Pipeline Execution ---
        # Stage A: Identify potential concepts using Bert1
        print("Running Stage A: Identifying concepts with Bert1...")
        concepts = bert1_detector.predict_links(text)
        if not concepts:
            print("Stage A did not find any concepts.")
            return {'concepts': {}}
        print(f"Stage A found {len(concepts)} potential concepts.")

        # Stage B: Filter concepts by domain relevance
        print("Running Stage B: Filtering concepts...")
        filtered_expressions = content_filter.filter_expressions(text, concepts)
        if not filtered_expressions:
            print("Stage B filtered out all concepts.")
            return {'concepts': {}}
        print(f"Stage B filtered down to {len(filtered_expressions)} concepts.")

        # Stage C: Rank filtered concepts using the features-based model
        print("Running Stage C: Ranking concepts...")
        ranked_concepts = prerequisite_ranker.rank_expressions(filtered_expressions, text)
        print(f"Stage C ranked {len(ranked_concepts)} concepts.")

        # --- Format for Frontend ---
        # Format all concepts as a list of objects for the frontend
        frontend_concepts = []
        for phrase, rank in ranked_concepts.items():
            frontend_concepts.append({
                "phrase": phrase,
                "rank": rank
            })
        
        # Sort by rank for better presentation
        frontend_concepts.sort(key=lambda x: x['rank'], reverse=True)

        print(f"Returning {len(frontend_concepts)} concepts to the frontend.")
        return {'concepts': frontend_concepts}

Handler = CustomHTTPRequestHandler

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

with ReusableTCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    print(f"Open http://localhost:{PORT} in your browser.")
    httpd.serve_forever()
