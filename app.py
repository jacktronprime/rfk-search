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
from bs4 import BeautifulSoup

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

# --- Setup Directories (only for indexes) ---
KEYWORD_INDEX_DIR = "index/keyword"
SEMANTIC_INDEX_DIR = "index/semantic"
os.makedirs(KEYWORD_INDEX_DIR, exist_ok=True)
os.makedirs(SEMANTIC_INDEX_DIR, exist_ok=True)

# --- Step 1: Scrape PDF URLs from National Archives ---
def scrape_pdf_urls():
    try:
        response = requests.get("https://www.archives.gov/research/rfk", timeout=10)
        if response.status_code != 200:
            logger.error(f"Failed to access National Archives page: Status {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a")
        logger.info(f"Found {len(links)} total links on the page")
        
        pdf_urls = []
        for link in links:
            href = link.get("href")
            if not href:
                logger.debug("Link without href attribute encountered")
                continue
            
            href = href.strip()
            if not href:
                continue
                
            if href.endswith(".pdf") and "rfk" in href.lower():
                if href.startswith("https://"):
                    pdf_urls.append(href)
                else:
                    pdf_urls.append(f"https://www.archives.gov{href}")
        
        logger.info(f"Found {len(pdf_urls)} PDF URLs")
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
        
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            text_chunks = []
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    text_chunks.append({"page": i + 1, "text": text})
                    logger.info(f"Page {i+1} (text): {text[:50]}...")
                elif PYTESSERACT_AVAILABLE:
                    try:
                        image = page.to_image(resolution=300).original
                        text = pytesseract.image_to_string(image, lang='eng')
                        if text.strip():
                            text_chunks.append({"page": i + 1, "text": text})
                            logger.info(f"Page {i+1} (OCR): {text[:50]}...")
                        else:
                            logger.warning(f"Page {i+1}: No text extracted via OCR")
                    except Exception as e:
                        logger.warning(f"OCR failed on page {i+1}: {e}")
                else:
                    logger.warning(f"Page {i+1}: No text extracted (OCR unavailable)")
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
                for chunk in chunks:
                    documents.append({
                        "filename": metadata["filename"],
                        "page": chunk["page"],
                        "text": chunk["text"],
                        "page_count": metadata["page_count"],
                        "url": metadata["url"]
                    })
        except Exception as e:
            logger.error(f"Error checking {url}: {e}")
    if not documents:
        logger.warning("No documents processed. Check URLs, network, or text extraction.")
        return pd.DataFrame()
    df = pd.DataFrame(documents)
    df.to_csv("documents_metadata.csv", index=False)
    logger.info(f"Processed {len(df)} document chunks from {len(urls)} PDFs")
    return df

# --- Scrape and Process URLs (Moved to Top Level) ---
# Scrape URLs
PDF_URLS = scrape_pdf_urls()

# Fallback: If scraping fails, use a few known URLs from the website content
if not PDF_URLS:
    logger.warning("Scraping failed. Using fallback URLs.")
    PDF_URLS = [
        "https://www.archives.gov/files/research/jfk/rfk/44-bh-1772-part-1-of-2.pdf",
        "https://www.archives.gov/files/research/jfk/rfk/166-12c-1-serial-1-56-la-156-la-report-6-15-68-part-1-of-7.pdf",
        "https://www.archives.gov/files/research/jfk/rfk/44-bh-1772-part-2-of-2.pdf",
    ]

# Process PDFs in batches to manage memory
BATCH_SIZE = 5
documents_df = pd.DataFrame()
if PDF_URLS:
    for i in range(0, len(PDF_URLS), BATCH_SIZE):
        batch_urls = PDF_URLS[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1} of {len(PDF_URLS)//BATCH_SIZE + 1}")
        batch_df = process_urls(batch_urls)
        documents_df = pd.concat([documents_df, batch_df], ignore_index=True)
else:
    logger.warning("No PDF URLs found to process.")

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

# Create keyword index if documents exist
if os.path.exists(KEYWORD_INDEX_DIR) and os.listdir(KEYWORD_INDEX_DIR):
    logger.info("Keyword index found, skipping creation.")
else:
    if not documents_df.empty:
        create_keyword_index(documents_df, KEYWORD_INDEX_DIR)
    else:
        logger.warning("Skipping keyword index creation due to empty document set.")

# --- Step 5: Create Semantic Index with Sentence Transformers ---
def create_semantic_index(documents_df):
    if documents_df.empty:
        logger.warning("No documents to index for semantic search.")
        return
    try:
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
        logger.info("Semantic index created")
    except Exception as e:
        logger.error(f"Error creating semantic index: {e}")

# Create semantic index if documents exist
if os.path.exists(SEMANTIC_INDEX_DIR) and os.path.exists(os.path.join(SEMANTIC_INDEX_DIR, "index.pkl")):
    logger.info("Semantic index found, skipping creation.")
else:
    if not documents_df.empty:
        create_semantic_index(documents_df)
    else:
        logger.warning("Skipping semantic index creation due to empty document set.")

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
        model = SentenceTransformer("all-MiniLM-L6-v2")
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
                "content": ""
            }
            for i in top_indices
        ]
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return []

# --- Step 7: Flask Web Interface ---
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    query = ""
    error_message = ""
    status_message = f"Indexed {len(documents_df)} pages from {len(PDF_URLS)} PDFs."
    
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
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)