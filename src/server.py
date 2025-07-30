
import os
import sys
import json
import http.server
import socketserver
import tempfile
import torch
from werkzeug.utils import secure_filename
from email.parser import BytesParser
from email.policy import default
from transformers import AutoTokenizer, AutoModel

# --- Add src to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from util.pdf_to_txt import pdf_to_clean_text
from stage_a.LinkDetector import KnowFlowLinkDetector
from stage_b.filter import ContentDomainFilter
from stage_c.prerequisite_extractor_encoder import PrerequisiteRankerEncoder

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
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- Model Loading ---
print("Loading pipeline models...")

# Shared BERT model
print("Loading shared BERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device)


# Stage A
print("Loading Stage A: Link Detector...")
LINK_DETECTOR_MODEL_PATH = 'models/link_detector.pt'
link_detector = KnowFlowLinkDetector(
    model_path=LINK_DETECTOR_MODEL_PATH,
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
print("Loading Stage C: Prerequisite Ranker...")
RANKER_MODEL_PATH = 'models/stage_c_ranker_encoder_penalty.pt'
prerequisite_ranker = PrerequisiteRankerEncoder(model_path=RANKER_MODEL_PATH)

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/templates/index.html'
        elif self.path.startswith('/results/'):
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path != '/upload':
            self.send_response(404)
            self.end_headers()
            return

        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Expected multipart/form-data')
            return

        boundary = content_type.split('boundary=')[-1].encode()
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        parts = body.split(b'--' + boundary)
        for part in parts:
            if b'Content-Disposition' not in part or b'filename=' not in part:
                continue

            try:
                headers, file_data = part.split(b'\r\n\r\n', 1)
                file_data = file_data.rstrip(b'\r\n--')

                headers_str = headers.decode()
                disposition_line = [line for line in headers_str.split('\r\n') if 'filename=' in line][0]
                filename = disposition_line.split('filename=')[1].strip().strip('"')
                filename = secure_filename(filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                with open(filepath, 'wb') as f:
                    f.write(file_data)

                print(f"Received file: {filename}")
                response_data = self.process_file(filepath, filename)

                # Save to a .txt file for download
                result_filename = os.path.splitext(filename)[0] + '_results.txt'
                result_path = os.path.join(RESULT_FOLDER, result_filename)
                with open(result_path, 'w', encoding='utf-8') as rf:
                    for item in response_data['concepts']:
                        rf.write(f"{item['phrase']}\tRank: {item['rank']}\n")

                # Return both the concepts for display and a download link
                response_payload = {
                    'concepts': response_data['concepts'],
                    'download_url': f'/results/{result_filename}'
                }

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
                os.remove(filepath)
                return

            except Exception as e:
                print("Upload error:", e)
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
                return

        self.send_response(400)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': 'No file found in upload'}).encode('utf-8'))

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

        print("Running Stage A: Identifying concepts with LinkDetector...")
        concepts = link_detector.predict_links(text)
        if not concepts:
            print("Stage A did not find any concepts.")
            return {'concepts': []}

        print("Running Stage B: Filtering concepts...")
        filtered_expressions = content_filter.filter_expressions(text, concepts)
        if not filtered_expressions:
            print("Stage B filtered out all concepts.")
            return {'concepts': []}

        print("Running Stage C: Ranking concepts...")
        ranked_concepts = prerequisite_ranker.rank_expressions(filtered_expressions, text)
        frontend_concepts = [
            {"phrase": phrase, "rank": rank}
            for phrase, rank in ranked_concepts.items()
        ]
        frontend_concepts.sort(key=lambda x: x['rank'], reverse=True)
        return {'concepts': frontend_concepts}

Handler = CustomHTTPRequestHandler

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

with ReusableTCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    print(f"Open http://localhost:{PORT} in your browser.")
    httpd.serve_forever()
