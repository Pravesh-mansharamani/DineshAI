"""
Telegram bot for answering Flare Network questions using RAG-enhanced consensus.
This bot combines multiple AI models with RAG and consensus for reliable answers.
"""

import os
import json
import logging
import asyncio
import time
from typing import Any

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from dotenv import load_dotenv

# Import the integrated RAG-Consensus engine
from rag_consensus_integration import RagConsensusEngine
from flare_ai_consensus.consensus_engine import ModelAPIClient

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Telegram bot token from environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

# Initialize global variables
rag_consensus_engine = None

# Load consensus configuration
def load_config() -> Dict[str, Any]:
    """Load the consensus engine configuration from input.json."""
    try:
        with open("src/flare_ai_consensus/input.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error("Consensus configuration file not found!")
        return {}

async def get_answer(prompt: str) -> str:
    """
    Get an answer from the RAG-Consensus engine.
    
    Args:
        prompt: The user's message/question
        
    Returns:
        Verified answer from the RAG-Consensus engine
    """
    global rag_consensus_engine
    
    try:
        # Get answer from the RAG-Consensus engine
        if rag_consensus_engine:
            result = await rag_consensus_engine.answer_question(prompt, complexity_level="auto")
            return result
        else:
            return "Sorry, the answer system is still initializing. Please try again in a moment."
    except Exception as e:
        logger.error(f"Error getting answer: {e}")
        return f"Sorry, I encountered an error while processing your request: {str(e)}"

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    await update.message.reply_text(
        "Welcome to the Flare Network Assistant! 👋\n\n"
        "I can answer your questions about Flare Network using multiple AI models "
        "with our knowledge base to provide accurate information.\n\n"
        "Example questions:\n"
        "• How does Flare Network achieve consensus?\n"
        "• What are the key features of Flare?\n"
        "• How do I become a validator?\n\n"
        "Just ask me anything about Flare!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
    await update.message.reply_text(
        "How to use the Flare Network Assistant:\n\n"
        "1. Ask any question about Flare Network\n"
        "2. Wait a moment while I analyze your question using multiple AI models\n"
        "3. Receive a comprehensive answer based on our documentation\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/about - Learn about how this bot works\n"
        "/index - Reindex the knowledge base (admin only)\n\n"
        "For technical support, please contact the developers."
    )

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send information about how the bot works."""
    await update.message.reply_text(
        "About Flare Network Assistant 🔍\n\n"
        "This bot uses advanced AI technology to provide accurate information about Flare Network:\n\n"
        "• RAG (Retrieval Augmented Generation): Your questions are used to retrieve relevant information from Flare's documentation\n\n"
        "• Multi-model consensus: Different AI models analyze the documentation and generate responses\n\n"
        "• Aggregation: The responses are combined into a single, accurate answer\n\n"
        "This approach ensures that you receive high-quality, factual information about Flare Network."
    )

async def reindex_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reindex the knowledge base when the command /index is issued."""
    # Only allow specific users to reindex (could be enhanced with proper admin controls)
    allowed_users = [12345678]  # Replace with actual admin user IDs
    current_user_id = update.effective_user.id
    
    # For testing, always allow reindexing
    if current_user_id not in allowed_users:
        await update.message.reply_text("For now, anyone can reindex during development.")
    
    await update.message.reply_text("Starting to reindex the knowledge base. This may take some time...")
    
    try:
        global rag_consensus_engine
        # Ensure RAG-Consensus engine is initialized
        if not rag_consensus_engine:
            await update.message.reply_text("❌ RAG-Consensus engine is not initialized.")
            return
        
        # Process documents
        success = rag_consensus_engine.index_documents("docs")
        
        if success:
            await update.message.reply_text("✅ Knowledge base reindexed successfully!")
        else:
            await update.message.reply_text("❌ Error reindexing knowledge base.")
    except Exception as e:
        logger.error(f"Error reindexing knowledge base: {e}")
        await update.message.reply_text(f"❌ Error reindexing knowledge base: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the user message and get a response using RAG-Consensus."""
    message_text = update.message.text
    
    # Send a typing action to indicate the bot is processing
    await update.message.chat.send_action(action="typing")
    
    # Get start time for timing the response
    start_time = time.time()
    
    # Get the consensus answer
    response = await get_answer(message_text)
    
    # Calculate response time
    response_time = time.time() - start_time
    logger.info(f"Response time: {response_time:.2f} seconds")
    
    # Send the response
    await update.message.reply_text(response)

async def initialize_rag_system():
    """Initialize the RAG system."""
    global rag_consensus_engine
    
    try:
        logger.info("Initializing RAG system...")
        
        # Load configuration
        config = load_config()
        if not config:
            logger.error("Failed to load configuration")
            return
        
        # Initialize API client if we have a key
        api_client = None
        if OPEN_ROUTER_API_KEY:
            api_client = ModelAPIClient(
                api_key=OPEN_ROUTER_API_KEY,
                base_url="https://openrouter.ai/api"
            )
            logger.info("API client initialized successfully")
        else:
            logger.warning("OpenRouter API key not found. Models will use default settings.")
        
        # Initialize the integrated RAG-Consensus engine
        rag_consensus_engine = RagConsensusEngine(
            config=config,
            embedding_dir="faiss_index",
            api_client=api_client
        )
        
        # Check if FAISS index exists and create if needed
        if not os.path.exists(os.path.join("faiss_index", "index.faiss")):
            logger.info("FAISS index not found. Creating index...")
            rag_consensus_engine.index_documents("docs")
        
        logger.info("✅ RAG system initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        logger.warning("Bot will run with limited capabilities.")

def main() -> None:
    """Start the bot."""
    if not TOKEN:
        logger.error("Telegram bot token not found. Please set TELEGRAM_BOT_TOKEN environment variable.")
        return
    
    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()

    # Initialize the RAG system
    asyncio.get_event_loop().run_until_complete(initialize_rag_system())

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("index", reindex_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until the user presses Ctrl-C
    logger.info("Bot started! Press Ctrl+C to stop.")
    application.run_polling()

if __name__ == "__main__":
    main()