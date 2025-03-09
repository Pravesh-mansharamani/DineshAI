import os
import logging
from typing import Any
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import numpy as np

from utils.file_utils import (
    get_files_info, load_ignore_patterns, calculate_chunks
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Google API
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)

def get_token_encoder():
    """Get the token encoder for OpenAI models."""
    return tiktoken.get_encoding("cl100k_base")

def encode_text(text: str) -> int:
    """
    Encode text and return token count.
    
    Args:
        text: Text to encode
        
    Returns:
        Token count
    """
    encoder = get_token_encoder()
    return len(encoder.encode(text))

class HybridEmbeddings:
    """
    Hybrid embedding model that combines OpenAI and Google embeddings.
    """
    
    def __init__(self):
        """Initialize the hybrid embedding system."""
        self.openai_embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            model="text-embedding-3-small"
        )
        
        self.google_embeddings = None
        if google_api_key:
            try:
                self.google_embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=google_api_key
                )
                logger.info("Google embeddings initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Google embeddings: {e}")
                logger.warning("Continuing with OpenAI embeddings only")
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents using both OpenAI and Google embeddings if available.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embeddings
        """
        # Get OpenAI embeddings
        openai_result = self.openai_embeddings.embed_documents(texts)
        
        # If Google embeddings are available, combine them
        if self.google_embeddings:
            try:
                google_result = self.google_embeddings.embed_documents(texts)
                
                # Normalize and concatenate embeddings
                combined_embeddings = []
                for openai_emb, google_emb in zip(openai_result, google_result):
                    # Normalize
                    openai_norm = np.array(openai_emb) / np.linalg.norm(openai_emb)
                    google_norm = np.array(google_emb) / np.linalg.norm(google_emb)
                    
                    # Concatenate and normalize again
                    combined = np.concatenate([openai_norm, google_norm])
                    combined = combined / np.linalg.norm(combined)
                    
                    combined_embeddings.append(combined.tolist())
                
                return combined_embeddings
            except Exception as e:
                logger.warning(f"Error combining embeddings: {e}")
                logger.warning("Falling back to OpenAI embeddings only")
        
        return openai_result
    
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a query text using both OpenAI and Google embeddings if available.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        # Get OpenAI embeddings
        openai_result = self.openai_embeddings.embed_query(text)
        
        # If Google embeddings are available, combine them
        if self.google_embeddings:
            try:
                google_result = self.google_embeddings.embed_query(text)
                
                # Normalize and concatenate embeddings
                openai_norm = np.array(openai_result) / np.linalg.norm(openai_result)
                google_norm = np.array(google_result) / np.linalg.norm(google_result)
                
                # Concatenate and normalize again
                combined = np.concatenate([openai_norm, google_norm])
                combined = combined / np.linalg.norm(combined)
                
                return combined.tolist()
            except Exception as e:
                logger.warning(f"Error combining embeddings: {e}")
                logger.warning("Falling back to OpenAI embeddings only")
        
        return openai_result

class RAGProcessor:
    """Class for processing and embedding documents for RAG."""
    
    def __init__(self, embedding_dir: str = "faiss_index"):
        """
        Initialize the RAG processor.
        
        Args:
            embedding_dir: Directory to store FAISS index
        """
        self.embedding_dir = embedding_dir
        self.hybrid_embeddings = HybridEmbeddings()
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(embedding_dir, exist_ok=True)
        
        logger.info("✅ RAG processor initialized!")
    
    def process_documents(self, directory: str, extensions: list[str] | None = None):
        """
        Process documents from a directory and store them in FAISS.
        
        Args:
            directory: Directory containing documents to process
            extensions: List of file extensions to include
        """
        if extensions is None:
            extensions = ['.md', '.mdx', '.py']
        
        # Load ignore patterns
        ignore_patterns = load_ignore_patterns()
        
        # Get file information
        logger.info(f"📁 Scanning for files with extensions: {extensions} in {directory}...")
        files_info = get_files_info(directory, extensions, ignore_patterns)
        
        if not files_info:
            logger.warning(f"⚠️ No files found with extensions {extensions} in {directory}")
            return
        
        logger.info(f"📝 Found {len(files_info)} files to process")
        
        # Process each file and collect chunks
        all_chunks = []
        all_metadatas = []
        
        success_count = 0
        error_count = 0
        
        for file_info in files_info:
            try:
                file_path = file_info['file_path']
                token_count = file_info['token_count']
                
                logger.info(f"📄 Processing {file_path} ({token_count} tokens)")
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Calculate chunks
                chunk_info = calculate_chunks(token_count)
                
                if chunk_info["num_chunks"] == 1:
                    # Add as a single document
                    logger.info(f"Adding {file_path} as a single document")
                    all_chunks.append(content)
                    all_metadatas.append({"source": file_path})
                else:
                    # Split into chunks
                    logger.info(f"Splitting {file_path} into {chunk_info['num_chunks']} chunks")
                    
                    # Create text splitter with appropriate chunk size
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_info["target_size"] * 4,  # Approximate ratio of characters to tokens
                        chunk_overlap=int(chunk_info["target_size"] * 0.1 * 4),  # 10% overlap
                        length_function=encode_text
                    )
                    
                    # Split the document
                    chunks = splitter.split_text(content)
                    
                    # Add chunk token count
                    chunk_tokens = [encode_text(chunk) for chunk in chunks]
                    logger.info(f"Created {len(chunks)} chunks with token counts: {chunk_tokens}")
                    
                    # Check for large chunks
                    large_chunks = [i for i, tokens in enumerate(chunk_tokens) if tokens > 800]
                    if large_chunks:
                        logger.warning(f"⚠️ {len(large_chunks)} chunks are larger than 800 tokens: {[chunk_tokens[i] for i in large_chunks]}")
                    
                    # Add chunks and metadata
                    all_chunks.extend(chunks)
                    all_metadatas.extend([{"source": file_path, "chunk": i} for i in range(len(chunks))])
                
                success_count += 1
                logger.info(f"✅ Successfully processed {file_path}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"❌ Error processing file {file_info['file_path']}: {str(e)}")
        
        # Create FAISS index from all chunks
        if all_chunks:
            logger.info(f"Creating FAISS index with {len(all_chunks)} chunks...")
            
            # Create FAISS vector store
            vector_store = FAISS.from_texts(
                texts=all_chunks,
                embedding=self.hybrid_embeddings,
                metadatas=all_metadatas
            )
            
            # Save the index
            vector_store.save_local(self.embedding_dir)
            logger.info(f"✅ FAISS index saved to {self.embedding_dir}")
        
        # Summary
        logger.info("\n📊 Processing Summary:")
        logger.info(f"✅ Successfully processed: {success_count} files")
        if error_count > 0:
            logger.info(f"❌ Failed to process: {error_count} files")
        logger.info("🎉 Processing complete!")
    
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search for documents similar to a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        try:
            logger.info(f"🔍 Searching for: {query}")
            
            # Check if index exists
            if not os.path.exists(os.path.join(self.embedding_dir, "index.faiss")):
                logger.warning("⚠️ FAISS index not found. Please process documents first.")
                return []
            
            # Load FAISS index
            vector_store = FAISS.load_local(self.embedding_dir, self.hybrid_embeddings)
            
            # Search
            results_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
            
            # Format results
            formatted_results = []
            for doc, score in results_with_scores:
                formatted_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk": doc.metadata.get("chunk", 0),
                    "similarity": float(score)  # Convert numpy float to Python float
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"❌ Error searching: {str(e)}")
            return []
    
    def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """
        Get relevant context for a query as a formatted string.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            Formatted context string
        """
        search_results = self.search(query, top_k=top_k)
        
        if not search_results:
            return ""
        
        # Format the context
        context_parts = []
        for i, result in enumerate(search_results):
            source = result["source"]
            content = result["content"]
            context_parts.append(f"[Document {i+1}] From {source}:\n{content}\n")
        
        return "\n".join(context_parts)

def process_embeddings(directory: str = "docs"):
    """
    Main function to process embeddings.
    
    Args:
        directory: Directory to process
    """
    try:
        logger.info("🚀 Starting RAG processor...")
        
        # Initialize RAG processor
        processor = RAGProcessor()
        
        # Process documents
        processor.process_documents(directory)
        
        logger.info("✅ RAG processor completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    process_embeddings() 