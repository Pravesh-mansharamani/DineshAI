"""
Enhanced integration between RAG and Consensus systems for the Flare AI bot.

This module provides a clean interface to combine the RAG (Retrieval Augmented Generation)
capabilities with the multi-model consensus approach. It handles document indexing,
retrieval, and integration with the consensus engine.
"""

import os
import logging
from typing import Any

# Import RAG processor, ensuring we use FAISS instead of Pinecone
try:
    from utils.rag_utils import RAGProcessor
except ImportError as e:
    # If there's an import error related to Pinecone, use a simplified version
    if "langchain_pinecone" in str(e):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain.vectorstores import FAISS
        import tiktoken
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Define a simplified RAGProcessor that only uses FAISS
        class RAGProcessor:
            """Simplified RAG processor using FAISS instead of Pinecone."""
            
            def __init__(self, embedding_dir="faiss_index"):
                self.embedding_dir = embedding_dir
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
                    model="text-embedding-3-small"
                )
                logger.info("✅ Simplified RAG processor initialized with FAISS!")
            
            def get_token_count(self, text):
                """Get token count for a text."""
                encoder = tiktoken.get_encoding("cl100k_base")
                return len(encoder.encode(text))
            
            def process_documents(self, directory, extensions=None):
                """Process documents and create FAISS index."""
                if extensions is None:
                    extensions = ['.md', '.mdx', '.py']
                
                # Find all files with the given extensions
                all_files = []
                for root, _, files in os.walk(directory):
                    for file in files:
                        if any(file.endswith(ext) for ext in extensions):
                            all_files.append(os.path.join(root, file))
                
                if not all_files:
                    logger.warning(f"No files found with extensions {extensions}")
                    return False
                
                logger.info(f"Found {len(all_files)} files to process")
                
                # Process files
                texts = []
                metadatas = []
                
                for file_path in all_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Split if too large
                        token_count = self.get_token_count(content)
                        if token_count > 1000:
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=4000,
                                chunk_overlap=400,
                                length_function=self.get_token_count
                            )
                            chunks = splitter.split_text(content)
                            texts.extend(chunks)
                            metadatas.extend([{"source": file_path, "chunk": i} for i in range(len(chunks))])
                        else:
                            texts.append(content)
                            metadatas.append({"source": file_path})
                        
                        logger.info(f"Processed {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                
                # Create FAISS index
                if texts:
                    vectorstore = FAISS.from_texts(
                        texts=texts,
                        embedding=self.embeddings,
                        metadatas=metadatas
                    )
                    
                    # Save the index
                    os.makedirs(self.embedding_dir, exist_ok=True)
                    vectorstore.save_local(self.embedding_dir)
                    logger.info(f"FAISS index saved to {self.embedding_dir}")
                    return True
                
                return False
            
            def get_relevant_context(self, query, top_k=3):
                """Get relevant context for a query."""
                try:
                    # Check if index exists
                    if not os.path.exists(os.path.join(self.embedding_dir, "index.faiss")):
                        logger.warning("FAISS index not found")
                        return ""
                    
                    # Load the index
                    vectorstore = FAISS.load_local(self.embedding_dir, self.embeddings)
                    
                    # Search
                    docs = vectorstore.similarity_search(query, k=top_k)
                    
                    # Format the context
                    if not docs:
                        return ""
                    
                    context_parts = []
                    for i, doc in enumerate(docs):
                        source = doc.metadata.get("source", "Unknown")
                        content = doc.page_content
                        context_parts.append(f"[Document {i+1}] From {source}:\n{content}\n")
                    
                    return "\n".join(context_parts)
                except Exception as e:
                    logger.error(f"Error retrieving context: {e}")
                    return ""
    else:
        # If it's some other import error, re-raise it
        raise
from flare_ai_consensus.consensus_engine import run_consensus

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

class RagConsensusEngine:
    """
    Integrates RAG capabilities with the Flare AI consensus system.
    
    This class provides methods to:
    1. Index documents for RAG retrieval
    2. Answer questions using RAG-enhanced consensus from multiple LLMs
    3. Manage the integration between vector search and consensus systems
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        embedding_dir: str = "faiss_index",
        api_client = None
    ):
        """
        Initialize the RAG Consensus Engine.
        
        Args:
            config: Consensus engine configuration
            embedding_dir: Directory for storing FAISS index
            api_client: Optional API client for model access
        """
        self.config = config
        self.api_client = api_client
        
        # Initialize RAG processor
        try:
            self.rag_processor = RAGProcessor(embedding_dir=embedding_dir)
            logger.info("✅ RAG processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG processor: {e}")
            raise
    
    def index_documents(self, directory: str, extensions: Optional[List[str]] = None) -> bool:
        """
        Index documents for later retrieval.
        
        Args:
            directory: Directory containing documents to index
            extensions: List of file extensions to include (default: .md, .mdx, .py)
            
        Returns:
            Success status (boolean)
        """
        try:
            logger.info(f"Indexing documents in {directory}...")
            self.rag_processor.process_documents(directory, extensions)
            logger.info("✅ Document indexing complete")
            return True
        except Exception as e:
            logger.error(f"❌ Error indexing documents: {e}")
            return False
    
    async def answer_question(self, question: str, complexity_level: str = "auto") -> str:
        """
        Answer a question using RAG-enhanced consensus.
        
        Args:
            question: The user's question
            complexity_level: Complexity level for response (simple, moderate, complex, auto)
            
        Returns:
            Consensus answer with RAG enhancement
        """
        try:
            # 1. Retrieve relevant context from indexed documents
            logger.info(f"Retrieving context for: {question}")
            context = self.rag_processor.get_relevant_context(question, top_k=3)
            
            if context:
                logger.info(f"Found relevant context ({len(context.split())} words)")
            else:
                logger.info("No relevant context found, proceeding with general knowledge")
            
            # 2. Run consensus with the retrieved context
            logger.info("Running consensus with RAG context")
            
            # Prepare a system message with context if available
            system_message = None
            if context:
                system_message = (
                    "You are answering questions about the Flare Network. "
                    "Use the following context to inform your response:\n\n"
                    f"{context}\n\n"
                    "If the context doesn't contain relevant information, use your general knowledge "
                    "but acknowledge the limitations. Always provide accurate information and "
                    "indicate clearly when information is not from the provided context."
                )
            
            # Run the consensus engine with context
            response = run_consensus(
                config=self.config,
                prompt=question,
                context=context,
                complexity_level=complexity_level,
                api_client=self.api_client,
                system_message=system_message
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"