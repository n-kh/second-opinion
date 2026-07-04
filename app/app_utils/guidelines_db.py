import os
import re
import logging
from typing import List, Dict, Any, Optional
import google.auth
from google.genai import Client

logger = logging.getLogger(__name__)

# Try importing chromadb; fallback gracefully if there are import issues
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".chroma_db"))
GUIDELINES_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "guidelines"))

def has_gcp_credentials() -> bool:
    if os.environ.get("INTEGRATION_TEST") == "TRUE":
        return False
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return True
    try:
        google.auth.default()
        return True
    except Exception:
        return False

def get_genai_client() -> Optional[Client]:
    if not has_gcp_credentials():
        return None
    try:
        # Client automatically loads environment credentials/API keys
        return Client()
    except Exception as e:
        logger.warning(f"Failed to initialize GenAI Client: {e}")
        return None

def get_embedding(text: str, client: Client) -> Optional[List[float]]:
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        if response.embeddings:
            return response.embeddings[0].values
    except Exception as e:
        logger.error(f"Error generating embedding for text: {e}")
    return None

def chunk_markdown(text: str) -> List[Dict[str, str]]:
    """
    Chunks a markdown guidelines file by headers or sections.
    """
    chunks = []
    # Split by headers (e.g. ## section)
    sections = re.split(r'\n(##\s+.*?)\n', text)
    
    # First section is introduction or title
    intro = sections[0].strip()
    if intro:
        chunks.append({
            "section": "General",
            "content": intro
        })
        
    for i in range(1, len(sections), 2):
        header = sections[i].replace("##", "").strip()
        content = sections[i+1].strip() if i+1 < len(sections) else ""
        if content:
            chunks.append({
                "section": header,
                "content": f"## {header}\n{content}"
            })
            
    return chunks

class GuidelinesVectorDB:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.genai_client = get_genai_client()
        
        if CHROMA_AVAILABLE:
            try:
                os.makedirs(DB_DIR, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(path=DB_DIR)
                # Create or get collection
                self.collection = self.chroma_client.get_or_create_collection(
                    name="nccn_guidelines"
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")

    def seed_database(self) -> bool:
        """
        Loads the markdown guidelines files, chunks them, embeds them,
        and saves them into the vector database.
        """
        if not self.collection:
            logger.warning("ChromaDB collection is not initialized.")
            return False
            
        client = self.genai_client or get_genai_client()
        if not client:
            logger.warning("GenAI Client not available for seeding (no credentials). Seeding skipped.")
            return False

        if not os.path.exists(GUIDELINES_DIR):
            logger.warning(f"Guidelines directory {GUIDELINES_DIR} does not exist.")
            return False

        files = [f for f in os.listdir(GUIDELINES_DIR) if f.endswith(".md")]
        if not files:
            logger.warning(f"No guideline markdown files found in {GUIDELINES_DIR}.")
            return False

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for filename in files:
            cancer_type = "breast" if "breast" in filename.lower() else "lung" if "lung" in filename.lower() else "other"
            filepath = os.path.join(GUIDELINES_DIR, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            chunks = chunk_markdown(content)
            for idx, chunk in enumerate(chunks):
                chunk_text = chunk["content"]
                embedding = get_embedding(chunk_text, client)
                
                if embedding:
                    doc_id = f"{cancer_type}_{chunk['section'].replace(' ', '_').lower()}_{idx}"
                    documents.append(chunk_text)
                    embeddings.append(embedding)
                    metadatas.append({
                        "filename": filename,
                        "cancer_type": cancer_type,
                        "section": chunk["section"]
                    })
                    ids.append(doc_id)

        if ids:
            try:
                # Upsert into the collection
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Successfully seeded {len(ids)} chunks into guidelines database.")
                return True
            except Exception as e:
                logger.error(f"Error upserting chunks to ChromaDB: {e}")
        
        return False

    def query_offline(self, query: str, cancer_type: Optional[str] = None) -> List[str]:
        """
        Fallback keyword matching search when offline / no GCP credentials.
        """
        results = []
        if not os.path.exists(GUIDELINES_DIR):
            return results
            
        files = [f for f in os.listdir(GUIDELINES_DIR) if f.endswith(".md")]
        for filename in files:
            file_cancer_type = "breast" if "breast" in filename.lower() else "lung" if "lung" in filename.lower() else "other"
            if cancer_type and file_cancer_type != cancer_type:
                continue
                
            filepath = os.path.join(GUIDELINES_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            chunks = chunk_markdown(content)
            for chunk in chunks:
                chunk_text = chunk["content"]
                # Basic keyword relevance scoring
                query_words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
                score = sum(1 for word in query_words if word in chunk_text.lower())
                
                # Check for direct terms
                if "her2" in query.lower() and "her2" in chunk_text.lower():
                    score += 5
                if "egfr" in query.lower() and "egfr" in chunk_text.lower():
                    score += 5
                    
                if score > 0:
                    results.append((score, chunk_text))
                    
        # Sort by relevance score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in results[:3]]

    def retrieve_guidelines(self, query: str, cancer_type: Optional[str] = None, limit: int = 3) -> List[str]:
        """
        Retrieves relevant NCCN guidelines matching the query.
        Uses vector RAG if credentials & database are ready, otherwise falls back to keyword matching.
        """
        # If we have no credentials, do a keyword matching offline query
        client = self.genai_client or get_genai_client()
        if not client or not self.collection or self.collection.count() == 0:
            logger.info("Using offline keyword search for guideline retrieval.")
            return self.query_offline(query, cancer_type)
            
        # Standardize cancer type label
        cancer_filter = None
        if cancer_type:
            cancer_filter = "breast" if "breast" in cancer_type.lower() else "lung" if "lung" in cancer_type.lower() else None
            
        query_embedding = get_embedding(query, client)
        if not query_embedding:
            logger.warning("Failed to embed query, using offline search.")
            return self.query_offline(query, cancer_type)
            
        try:
            where_clause = {}
            if cancer_filter:
                where_clause = {"cancer_type": cancer_filter}
                
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause if where_clause else None
            )
            
            if results and results.get("documents") and results["documents"][0]:
                return results["documents"][0]
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            
        return self.query_offline(query, cancer_type)

# Singleton guidelines DB instance
guidelines_db = GuidelinesVectorDB()
