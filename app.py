import os
import requests
import pdfplumber
import pandas as pd
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from flask import Flask, request, render_template_string
import logging
import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import pytesseract
import shutil
import psutil  # For disk and memory usage logging
from bs4 import BeautifulSoup
import gc  # For garbage collection

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Verify pytesseract and Tesseract ---
try:
    PYTESSERACT_AVAILABLE = True
    logger.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
    logger.info(f"Tesseract path: {shutil.which('tesseract')}")
except Exception as e:
    PYTESSERACT_AVAILABLE = False
    logger.warning(f"pytesseract/Tesseract unavailable: {e}. OCR will be skipped.")

# --- Setup Persistent Disk Directories ---
KEYWORD_INDEX_DIR = "/opt/render/project/disk/index/keyword"
SEMANTIC_INDEX_DIR = "/opt/render/project/disk/index/semantic"
TEMP_DIR = "/opt/render/project/disk/tmp"
TESSDATA_DIR = "/opt/render/project/disk/tessdata"
METADATA_PATH = "/opt/render/project/disk/documents_metadata.csv"

os.makedirs(KEYWORD_INDEX_DIR, exist_ok=True)
os.makedirs(SEMANTIC_INDEX_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TESSDATA_DIR, exist_ok=True)
os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

# Global variables
PDF_URLS = None
documents_df = pd.DataFrame()

# --- Utility to Log Memory and Disk Usage ---
def log_resource_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    try:
        disk_usage = psutil.disk_usage("/opt/render/project/disk")
        logger.info(f"Disk usage: {disk_usage.used / 1024 / 1024:.2f} MB / {disk_usage.total / 1024 / 1024:.2f} MB")
    except Exception as e:
        logger.error(f"Disk usage check failed: {e}")

# --- Clean Up Temporary Files ---
def cleanup_temp_files():
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
        logger.info("Cleaned up temporary files")
    except Exception as e:
        logger.error(f"Failed to clean up temporary files: {e}")

# --- Step 1: Scrape PDF URLs from National Archives ---
def scrape_pdf_urls():
    try:
        response = requests.get("https://www.archives.gov/research/rfk", timeout=10)
        if response.status_code != 200:
            logger.error(f"Failed to access National Archives page: Status {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a")
        logger.info(f"Found {len(links)} total links")
        
        pdf_urls = []
        for link in links:
            href = link.get("href")
            if not href:
                continue
            href = href.strip()
            if href.endswith(".pdf") and "rfk" in href.lower():
                if href.startswith("https://"):
                    pdf_urls.append(href)
                else:
                    pdf_urls.append(f"https://www.archives.gov{href}")
        
        # Filter to relevant PDFs to reduce storage
        pdf_urls = [url for url in pdf_urls if "sirhan" in url.lower() or "fbi" in url.lower()]
        logger.info(f"Filtered to {len(pdf_urls)} relevant PDF URLs")
        if pdf_urls:
            logger.debug(f"Sample URLs: {pdf_urls[:3]}")
        return pdf_urls
    except Exception as e:
        logger.error(f"Error scraping PDF URLs: {e}")
        return []

# --- Step 2: Stream and Extract Text from PDFs ---
def stream_and_extract_text(url):
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1)
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.get(url, stream=True, timeout=30)
        if response.status_code != 200:
            logger.error(f"Failed to fetch {url}: Status {response.status_code}")
            return None, []
        
        pdf_data = io.BytesIO(response.content)
        text_chunks = []
        page_count = 0
        
        with pdfplumber.open(pdf_data) as pdf:
            page_count = len(pdf.pages)
            logger.info(f"Processing PDF: {url} with {page_count} pages")
            
            for i in range(page_count):
                page = pdf.pages[i]
                logger.info(f"Processing page {i+1} of {page_count}")
                
                text = page.extract_text() or ""
                if text.strip():
                    logger.info(f"Page {i+1} extracted text: {text[:50]}...")
                    text_chunks.append({"page": i + 1, "text": text})
                elif PYTESSERACT_AVAILABLE:
                    try:
                        logger.info(f"Attempting OCR on page {i+1}...")
                        image = page.to_image(resolution=200).original
                        text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
                        if text.strip():
                            logger.info(f"Page {i+1} OCR successful: {text[:50]}...")
                            text_chunks.append({"page": i + 1, "text": text})
                        else:
                            logger.warning(f"Page {i+1}: No text extracted via OCR")
                        del image
                        gc.collect()
                    except Exception as e:
                        logger.warning(f"OCR failed on page {i+1}: {e}")
                else:
                    logger.warning(f"Page {i+1}: No text extracted (OCR unavailable)")
                
                del page
                gc.collect()
                cleanup_temp_files()  # Clean up after each page
                log_resource_usage()
        
        del pdf_data
        gc.collect()
        
        metadata = {
            "filename": os.path.basename(url),
            "page_count": page_count,
            "url": url
        }
        logger.info(f"Processed {url}: {page_count} pages, {len(text_chunks)} with text")
        return metadata, text_chunks
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return None, []

# --- Step 3: Process URLs ---
def process_urls(urls):
    documents = []
    for url in urls:
        try:
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1)
            session.mount("https://", HTTPAdapter(max_retries=retries))
            response = session.head(url, timeout=5)
            if response.status_code != 200:
                logger.error(f"Invalid URL {url}: Status {response.status_code}")
                continue
            metadata, chunks = stream_and_extract_text(url)
            if metadata:
                logger.info(f"Appending {len(chunks)} chunks from {url}")
                batch_df = pd.DataFrame([{
                    "filename": metadata["filename"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "page_count": metadata["page_count"],
                    "url": metadata["url"]
                } for chunk in chunks])
                batch_df.to_csv(f"/opt/render/project/disk/batch_{url.split('/')[-1]}.csv", index=False)
                documents.extend(batch_df.to_dict("records"))
            else:
                logger.warning(f"No metadata or chunks for {url}")
            log_resource_usage()
        except Exception as e:
            logger.error(f"Error checking {url}: {e}")
    if not documents:
        logger.warning("No documents processed")
        return pd.DataFrame()
    df = pd.DataFrame(documents)
    logger.info(f"Created DataFrame with {len(df)} rows")
    df.to_csv(METADATA_PATH, index=False)
    logger.info(f"Saved metadata to {METADATA_PATH}")
    return df

# --- Step 4: Create Keyword Index with Whoosh ---
def create_keyword_index(documents_df, index_dir):
    if documents_df.empty:
        logger.warning("No documents to index for keyword search")
        return
    try:
        schema = Schema(
            filename=ID(stored=True),
            page=ID(stored=True),
            url=ID(stored=True),
            content=TEXT(stored=True)
        )
        index = create_in(index_dir, schema)
        writer = index.writer(limitmb=512)  # Cap index size
        for _, row in documents_df.iterrows():
            writer.add_document(
                filename=row["filename"],
                page=str(row["page"]),
                url=row["url"],
                content=row["text"]
            )
        writer.commit()
        logger.info("Keyword index created")
        log_resource_usage()
    except Exception as e:
        logger.error(f"Error creating keyword index: {e}")
        raise

# --- Step 5: Create Semantic Index with Sentence Transformers ---
def create_semantic_index(documents_df):
    if documents_df.empty:
        logger.warning("No documents to index for semantic search")
        return
    try:
        logger.info("Loading sentence-transformers model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = documents_df["text"].tolist()
        embeddings = model.encode(texts, show_progress_bar=True)
        
        semantic_index = {
            "embeddings": embeddings,
            "documents": documents_df[["filename", "page", "url"]].to_dict("records")
        }
        index_path = os.path.join(SEMANTIC_INDEX_DIR, "index.pkl")
        with open(index_path, "wb") as f:
            pickle.dump(semantic_index, f)
        logger.info(f"Semantic index saved to {index_path}")
        log_resource_usage()
        del model, embeddings, texts, semantic_index
        gc.collect()
        log_resource_usage()
    except Exception as e:
        logger.error(f"Error creating semantic index: {e}")
        raise

# --- Initialize Data ---
def initialize_data():
    global PDF_URLS, documents_df
    logger.info("Starting initialize_data...")
    try:
        # Scrape PDFs
        logger.info("Scraping PDF URLs...")
        PDF_URLS = scrape_pdf_urls()
        
        if not PDF_URLS:
            logger.warning("Scraping failed. Using fallback URLs.")
            PDF_URLS = [
                "https://www.archives.gov/files/research/jfk/rfk/44-bh-1772-part-1-of-2.pdf",
                "https://www.archives.gov/files/research/jfk/rfk/166-12c-1-serial-1-56-la-156-la-report-6-15-68-part-1-of-7.pdf",
                "https://www.archives.gov/files/research/jfk/rfk/44-bh-1772-part-2-of-2.pdf",
            ]
        logger.info(f"PDF_URLS: {len(PDF_URLS)}")
        
        # Process PDFs in batches
        BATCH_SIZE = 1
        documents_df = pd.DataFrame()
        for i in range(0, len(PDF_URLS), BATCH_SIZE):
            batch_urls = PDF_URLS[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}")
            batch_df = process_urls(batch_urls)
            documents_df = pd.concat([documents_df, batch_df], ignore_index=True)
            logger.info(f"Updated documents_df with {len(documents_df)} rows")
            log_resource_usage()
        
        if documents_df.empty:
            logger.warning("No documents processed")
            raise ValueError("No documents to index")
        
        # Create indexes
        if os.path.exists(KEYWORD_INDEX_DIR) and os.listdir(KEYWORD_INDEX_DIR):
            logger.info("Keyword index found, skipping creation")
        else:
            logger.info("Creating keyword index...")
            create_keyword_index(documents_df, KEYWORD_INDEX_DIR)
        
        if os.path.exists(SEMANTIC_INDEX_DIR) and os.path.exists(os.path.join(SEMANTIC_INDEX_DIR, "index.pkl")):
            logger.info("Semantic index found, skipping creation")
        else:
            logger.info("Creating semantic index...")
            create_semantic_index(documents_df)
        
        logger.info("Initialization complete")
    except Exception as e:
        logger.error(f"Error in initialize_data: {str(e)}", exc_info=True)
        raise

# --- Step 6: Search Functions ---
def keyword_search(query_str, index_dir, limit=5):
    try:
        index = open_dir(index_dir)
        with index.searcher() as searcher:
            query = QueryParser("content", index.schema).parse(query_str)
            results = searcher.search(query, limit=limit)
            return [
                {"filename": hit["filename"], "page": hit["page"], "url": hit["url"], "content": hit["content"][:200]}
                for hit in results
            ]
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return []

def semantic_search(query_str, limit=5):
    try:
        logger.info("Loading sentence-transformers model for search...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        index_path = os.path.join(SEMANTIC_INDEX_DIR, "index.pkl")
        if not os.path.exists(index_path):
            logger.warning("Semantic index file not found")
            return []
        with open(index_path, "rb") as f:
            semantic_index = pickle.load(f)
        
        query_embedding = model.encode([query_str])[0]
        embeddings = semantic_index["embeddings"]
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        results = [
            {
                "filename": semantic_index["documents"][i]["filename"],
                "page": semantic_index["documents"][i]["page"],
                "url": semantic_index["documents"][i]["url"],
                "content": ""
            }
            for i in top_indices
        ]
        del model, query_embedding, embeddings, similarities, semantic_index
        gc.collect()
        log_resource_usage()
        return results
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return []

# --- Step 7: Flask Web Interface ---
app = Flask(__name__)

@app.route("/health")
def health():
    return "OK", 200

@app.route("/initialize", methods=["POST"])
def initialize():
    global PDF_URLS, documents_df
    try:
        if PDF_URLS is None or documents_df.empty:
            logger.info("Starting data initialization...")
            initialize_data()
            logger.info("Data initialization completed")
            return "Data initialized successfully.", 200
        logger.info("Data already initialized")
        return "Data already initialized.", 200
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}", exc_info=True)
        return f"Initialization failed: {str(e)}", 500

@app.route("/", methods=["GET", "POST"])
def search():
    global PDF_URLS, documents_df
    results = []
    query = ""
    error_message = ""
    status_message = f"Indexed {len(documents_df)} pages from {len(PDF_URLS) if PDF_URLS else 0} PDFs."
    
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            keyword_results = keyword_search(query, KEYWORD_INDEX_DIR)
            semantic_results = semantic_search(query)
            results = keyword_results + semantic_results
            if not results:
                error_message = "No results found. Ensure documents are indexed or try a different query."
        else:
            error_message = "Please enter a search query."
    
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RFK Assassination Files Search</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .error { color: red; }
                .status { color: green; }
                a { color: blue; }
            </style>
        </head>
        <body>
            <h1>RFK Assassination Files Search</h1>
            <p class="status">{{status_message}}</p>
            <form method="POST">
                <input type="text" name="query" value="{{query}}" placeholder="Enter search query (e.g., Sirhan Sirhan)">
                <input type="submit" value="Search">
            </form>
            <form method="POST" action="/initialize">
                <input type="submit" value="Initialize Data">
            </form>
            {% if error_message %}
                <p class="error">{{error_message}}</p>
            {% endif %}
            <h2>Results</h2>
            {% if results %}
                {% for result in results %}
                    <p>
                        <b>File:</b> {{result.filename}}  
                        <b>Page:</b> {{result.page}}  
                        <b>URL:</b> <a href="{{result.url}}" target="_blank">{{result.url}}</a>  
                        {% if result.content %}
                            <br><b>Content Preview:</b> {{result.content}}...
                        {% endif %}
                    </p>
                {% endfor %}
            {% else %}
                <p>No results found or no query entered. Try searching for terms like 'Sirhan Sirhan' or 'FBI investigation'.</p>
            {% endif %}
        </body>
        </html>
    """, query=query, results=results, error_message=error_message, status_message=status_message)

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting Flask app")
    log_resource_usage()
    app.run(debug=False, host="0.0.0.0", port=5000)