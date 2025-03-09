import os
import logging
from typing import Any
from dotenv import load_dotenv
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.utils.file_utils import (
    get_files_info, load_ignore_patterns, calculate_chunks
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate environment variables
def validate_env_vars():
    """Validate that required environment variables are present."""
    required_env_vars = [
        'OPEN_ROUTER_API_KEY',
        'PINECONE_API_KEY',
        'PINECONE_INDEX'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True


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


class EmbeddingProcessor:
    """Class for processing and embedding documents."""
    
    def __init__(self):
        """Initialize the embedding processor."""
        # Check environment variables
        if not validate_env_vars():
            raise EnvironmentError("Missing required environment variables")
        
        # Initialize Pinecone
        logger.info("🔄 Initializing Pinecone...")
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Get or create Pinecone index
        self.index_name = os.getenv("PINECONE_INDEX", "flare-docs")
        self._initialize_index()
        
        # Set up OpenAI embeddings
        logger.info("🔄 Setting up OpenAI embeddings...")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            model="text-embedding-3-small"
        )
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index=self.pinecone.Index(self.index_name),
            embedding=self.embeddings,
            text_key="text",
            namespace="flare-docs"
        )
        
        logger.info("✅ Embedding processor initialized!")

    def _initialize_index(self):
        """Initialize or get the Pinecone index."""
        # Check if index exists
        existing_indexes = [idx.name for idx in self.pinecone.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=1536,  # Dimensionality for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
            logger.info(f"✅ Created new Pinecone index: {self.index_name}")
        else:
            logger.info(f"Using existing Pinecone index: {self.index_name}")

    def process_documents(self, directory: str, extensions: list[str] = None):
        """
        Process documents from a directory and store them in Pinecone.
        
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
        
        # Process each file
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
                    self.vector_store.add_texts(
                        texts=[content],
                        metadatas=[{"source": file_path}]
                    )
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
                    
                    # Create metadata for each chunk
                    metadatas = [{"source": file_path, "chunk": i} for i in range(len(chunks))]
                    
                    # Add to vector store
                    self.vector_store.add_texts(
                        texts=chunks,
                        metadatas=metadatas
                    )
                
                success_count += 1
                logger.info(f"✅ Successfully processed {file_path}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"❌ Error processing file {file_info['file_path']}: {str(e)}")
        
        # Summary
        logger.info("\n📊 Processing Summary:")
        logger.info(f"✅ Successfully processed: {success_count} files")
        if error_count > 0:
            logger.info(f"❌ Failed to process: {error_count} files")
        logger.info("🎉 Processing complete!")

    def search(self, query: str, top_k: int = 5) -> List[dict]:
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
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk": doc.metadata.get("chunk", 0),
                    "similarity": score
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"❌ Error searching: {str(e)}")
            return []


def process_embeddings():
    """Main function to process embeddings."""
    try:
        logger.info("🚀 Starting embeddings processor...")
        
        # Initialize embedding processor
        processor = EmbeddingProcessor()
        
        # Process documents from the docs directory
        processor.process_documents("docs")
        
        logger.info("✅ Embedding processor completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    process_embeddings() 