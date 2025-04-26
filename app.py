import os
import requests
import pdfplumber
import pandas as pd
from whoosh.index import create_in, open_dir, exists_in
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
from bs4 import BeautifulSoup
from markupsafe import escape
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify pytesseract and Tesseract
try:
    PYTESSERACT_AVAILABLE = True
    logger.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
except Exception as e:
    PYTESSERACT_AVAILABLE = False
    logger.warning(f"pytesseract/Tesseract unavailable: {e}. OCR will be skipped.")

# --- Setup Directories (use Render persistent disk) ---
KEYWORD_INDEX_DIR = "/mnt/disk/index/keyword" if os.path.exists("/mnt/disk") else "index/keyword"
SEMANTIC_INDEX_DIR = "/mnt/disk/index/semantic" if os.path.exists("/mnt/disk") else "index/semantic"
os.makedirs(KEYWORD_INDEX_DIR, exist_ok=True)
os.makedirs(SEMANTIC_INDEX_DIR, exist_ok=True)

# --- Global Variables ---
PDF_URLS = []
documents_df = pd.DataFrame()

# --- Step 1: Scrape PDF URLs from National Archives ---
def scrape_pdf_urls():
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get("https://www.archives.gov/research/rfk", headers=headers, timeout=10)
        response.raise_for_status()
        logger.info(f"Successfully accessed National Archives: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Failed to access National Archives: {e}")
        return []

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a")
        logger.info(f"Found {len(links)} links")
        pdf_urls = []
        for link in links:
            href = link.get("href")
            if not href or not href.strip():
                continue
            if href.endswith(".pdf") and "rfk" in href.lower():
                full_url = href if href.startswith("https://") else f"https://www.archives.gov{href}"
                pdf_urls.append(full_url)
        pdf_urls = list(set(pdf_urls))  # Remove duplicates
        logger.info(f"Found {len(pdf_urls)} unique PDF URLs: {pdf_urls[:3]}")
        return pdf_urls
    except Exception as e:
        logger.error(f"Error parsing page: {e}")
        return []

# --- Step 2: Stream and Extract Text from PDFs ---
def stream_and_extract_text(url):
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1)
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        logger.info(f"Downloaded {url}")
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None, []

    try:
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            text_chunks = []
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                    if text.strip() and len(text) > 50:
                        text_chunks.append({"page": i + 1, "text": text})
                        logger.info(f"Page {i+1} (text): {text[:50]}...")
                    elif PYTESSERACT_AVAILABLE:
                        try:
                            image = page.to_image(resolution=300).original
                            text = pytesseract.image_to_string(image, lang='eng')
                            if text.strip() and len(text) > 50:
                                text_chunks.append({"page": i + 1, "text": text})
                                logger.info(f"Page {i+1} (OCR): {text[:50]}...")
                            else:
                                logger.warning(f"Page {i+1}: No meaningful text via OCR")
                        except Exception as e:
                            logger.warning(f"OCR failed on page {i+1}: {e}")
                    else:
                        logger.warning(f"Page {i+1}: No text extracted (OCR unavailable)")
                except Exception as e:
                    logger.warning(f"Error processing page {i+1}: {e}")
            metadata = {
                "filename": os.path.basename(url),
                "page_count": page_count,
                "url": url
            }
            logger.info(f"Processed {url}: {page_count} pages, {len(text_chunks)} with text")
            return metadata, text_chunks
    except Exception as e:
        logger.error(f"Error processing PDF {url}: {e}")
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
                for chunk in chunks:
                    documents.append({
                        "filename": metadata["filename"],
                        "page": chunk["page"],
                        "text": chunk["text"],
                        "page_count": metadata["page_count"],
                        "url": metadata["url"]
                    })
            time.sleep(1)  # Avoid rate limits
        except Exception as e:
            logger.error(f"Error checking {url}: {e}")
    if not documents:
        logger.warning("No documents processed. Check URLs, network, or text extraction.")
        return pd.DataFrame()
    df = pd.DataFrame(documents)
    df.to_csv("/mnt/disk/documents_metadata.csv" if os.path.exists("/mnt/disk") else "documents_metadata.csv", index=False)
    logger.info(f"Processed {len(df)} document chunks from {len(urls)} PDFs")
    return df

# --- Step 4: Create Keyword Index with Whoosh ---
def create_keyword_index(documents_df, index_dir):
    if documents_df.empty:
        logger.warning("No documents to index for keyword search.")
        return
    try:
        schema = Schema(
            filename=ID(stored=True),
            page=ID(stored=True),
            url=ID(stored=True),
            content=TEXT(stored=True)
        )
        index = create_in(index_dir, schema)
        writer = index.writer()
        for _, row in documents_df.iterrows():
            writer.add_document(
                filename=row["filename"],
                page=str(row["page"]),
                url=row["url"],
                content=row["text"]
            )
        writer.commit()
        logger.info("Keyword index created")
    except Exception as e:
        logger.error(f"Error creating keyword index: {e}")

# --- Step 5: Create Semantic Index with Sentence Transformers ---
def create_semantic_index(documents_df):
    if documents_df.empty:
        logger.warning("No documents to index for semantic search.")
        return
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        texts = documents_df["text"].tolist()
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            embeddings.extend(model.encode(texts[i:i + batch_size], show_progress_bar=True))
        embeddings = np.array(embeddings)
        semantic_index = {
            "embeddings": embeddings,
            "documents": documents_df[["filename", "page", "url", "text"]].to_dict("records")
        }
        index_path = os.path.join(SEMANTIC_INDEX_DIR, "index.pkl")
        with open(index_path, "wb") as f:
            pickle.dump(semantic_index, f)
        logger.info("Semantic index created")
    except Exception as e:
        logger.error(f"Error creating semantic index: {e}")

# --- Step 6: Initialize Indices ---
def initialize_indices():
    global PDF_URLS, documents_df  # Declare as global to modify
    if exists_in(KEYWORD_INDEX_DIR) and os.path.exists(os.path.join(SEMANTIC_INDEX_DIR, "index.pkl")):
        logger.info("Existing indices found, loading metadata.")
        metadata_path = "/mnt/disk/documents_metadata.csv" if os.path.exists("/mnt/disk") else "documents_metadata.csv"
        documents_df = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else pd.DataFrame()
        PDF_URLS.extend(scrape_pdf_urls())  # Repopulate PDF_URLS if loading indices
        if not PDF_URLS:
            logger.warning("Scraping failed. Using fallback URLs.")
            PDF_URLS.extend([
                "https://www.archives.gov/files/research/jfk/rfk/44-bh-1772-part-1-of-2.pdf",
                "https://www.archives.gov/files/research/jfk/rfk/44-bh-1772-part-2-of-2.pdf",
            ])
        return
    PDF_URLS.extend(scrape_pdf_urls())
    if not PDF_URLS:
        logger.warning("Scraping failed. Using fallback URLs.")
        PDF_URLS.extend([
            "https://www.archives.gov/files/research/jfk/rfk/44-bh-1772-part-1-of-2.pdf",
            "https://www.archives.gov/files/research/jfk/rfk/44-bh-1772-part-2-of-2.pdf",
        ])
    BATCH_SIZE = 2
    documents_df = pd.DataFrame()
    for i in range(0, len(PDF_URLS), BATCH_SIZE):
        batch_urls = PDF_URLS[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1} of {len(PDF_URLS)//BATCH_SIZE + 1}")
        batch_df = process_urls(batch_urls)
        documents_df = pd.concat([documents_df, batch_df], ignore_index=True)
    if not documents_df.empty:
        create_keyword_index(documents_df, KEYWORD_INDEX_DIR)
        create_semantic_index(documents_df)
    else:
        logger.error("No documents processed.")

# --- Step 7: Search Functions ---
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
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        index_path = os.path.join(SEMANTIC_INDEX_DIR, "index.pkl")
        if not os.path.exists(index_path):
            logger.warning("Semantic index file not found.")
            return []
        with open(index_path, "rb") as f:
            semantic_index = pickle.load(f)
        query_embedding = model.encode([query_str])[0]
        embeddings = semantic_index["embeddings"]
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[::-1][:limit]
        return [
            {
                "filename": semantic_index["documents"][i]["filename"],
                "page": semantic_index["documents"][i]["page"],
                "url": semantic_index["documents"][i]["url"],
                "content": semantic_index["documents"][i]["text"][:200]
            }
            for i in top_indices
        ]
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return []

# --- Step 8: Flask Web Interface ---
app = Flask(__name__)

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Server error: {error}")
    return render_template_string("<h1>Server Error</h1><p>Something went wrong. Please try again later.</p>"), 500

@app.route("/", methods=["GET", "POST"])
def search():
    global PDF_URLS, documents_df  # Access global variables
    results = []
    query = ""
    error_message = ""
    status_message = f"Indexed {len(documents_df)} pages from {len(PDF_URLS)} PDFs."
    if request.method == "POST":
        query = escape(request.form.get("query", "").strip())
        if query:
            keyword_results = keyword_search(query, KEYWORD_INDEX_DIR)
            semantic_results = semantic_search(query)
            seen = set()
            for r in keyword_results + semantic_results:
                key = (r["filename"], r["page"])
                if key not in seen:
                    seen.add(key)
                    results.append(r)
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
    initialize_indices()
    logger.info("Starting Flask app")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))