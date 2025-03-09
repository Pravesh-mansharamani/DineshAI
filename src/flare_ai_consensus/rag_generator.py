import os
import logging
from typing import Any

from src.utils.rag_utils import RAGProcessor

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

class FlareRetrievalAugmentedGenerator:
    """
    Enhanced RAG generator for the Flare AI Consensus system.
    This class encapsulates retrieval functionality to provide context from
    documentation for the consensus engine.
    """
    
    def __init__(self, embedding_dir: str = "faiss_index"):
        """
        Initialize the RAG generator.
        
        Args:
            embedding_dir: Directory with FAISS index
        """
        try:
            self.rag_processor = RAGProcessor(embedding_dir=embedding_dir)
            logger.info("✅ FlareRetrievalAugmentedGenerator initialized")
        except Exception as e:
            logger.error(f"Error initializing RAG generator: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            top_k: Number of top results to retrieve
            
        Returns:
            Formatted context string
        """
        try:
            logger.info(f"Retrieving context for query: {query[:50]}...")
            
            # Get context from RAG processor
            context = self.rag_processor.get_relevant_context(query, top_k=top_k)
            
            if context:
                logger.info(f"✅ Found context ({len(context.split())} words)")
                return context
            else:
                logger.warning("⚠️ No relevant context found")
                return ""
                
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""
    
    def index_documents(self, directory: str = "docs", extensions: list[str] | None = None) -> bool:
        """
        Index documents for retrieval.
        
        Args:
            directory: Directory with documents
            extensions: File extensions to process
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Indexing documents in {directory}...")
            self.rag_processor.process_documents(directory, extensions)
            logger.info("✅ Indexing complete")
            return True
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False 