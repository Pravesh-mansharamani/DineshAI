import os
import json
import logging
from typing import Any
import asyncio

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from dotenv import load_dotenv

# Import the consensus engine
from flare_ai_consensus.consensus_engine import run_consensus

# Import our RAG utilities
from src.utils.embedding_processor import EmbeddingProcessor

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Telegram bot token from environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Initialize the embedding processor for RAG
embedding_processor = None

# Load consensus configuration
def load_config() -> dict[str, Any]:
    """Load the consensus engine configuration from input.json."""
    try:
        with open("src/flare_ai_consensus/input.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error("Consensus configuration file not found!")
        return {}

# A wrapper function for the consensus engine with RAG
async def get_consensus_answer(prompt: str) -> str:
    """
    Get a consensus answer from the flare_ai_consensus module with RAG enhancement.
    
    Args:
        prompt: The user's message/question to verify
        
    Returns:
        A verified answer from the consensus engine
    """
    config = load_config()
    
    try:
        # Get relevant context from the knowledge base
        context = ""
        if embedding_processor:
            search_results = embedding_processor.search(prompt, top_k=3)
            if search_results:
                context = "\n\n".join([res["content"] for res in search_results])
                logger.info(f"Found {len(search_results)} relevant documents for RAG")
        
        # Create a prompt with relevant context
        if context:
            enhanced_prompt = (
                f"Based on the following context from our knowledge base, "
                f"please verify the following claim or answer the following question with factual information. "
                f"Mark certainty levels explicitly (VERIFIED, LIKELY, UNCERTAIN).\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"CLAIM/QUESTION: {prompt}"
            )
        else:
            enhanced_prompt = (
                f"Please verify the following claim or answer the following question with factual information. "
                f"Mark certainty levels explicitly (VERIFIED, LIKELY, UNCERTAIN). "
                f"Claim/Question: {prompt}"
            )
        
        # Run the consensus engine
        result = run_consensus(config, enhanced_prompt)
        return result
    except Exception as e:
        logger.error(f"Error getting consensus answer: {e}")
        return f"Sorry, I encountered an error while processing your request: {str(e)}"

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    await update.message.reply_text(
        "Welcome to CapCheck - a community notes for Telegram with RAG capabilities!\n\n"
        "Send me any message, claim, or question about Flare, and I'll verify it using multiple AI models with our knowledge base to provide a consensus answer.\n\n"
        "Example: 'How does Flare network achieve consensus?' or 'What are the key features of Flare?'"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
    await update.message.reply_text(
        "How to use CapCheck:\n\n"
        "1. Send any claim or question about Flare you want to verify\n"
        "2. Wait a moment while multiple AI models analyze it against our knowledge base\n"
        "3. Receive a consensus answer with certainty levels marked as VERIFIED, LIKELY, or UNCERTAIN\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/index - Reindex the knowledge base (admin only)"
    )

async def reindex_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reindex the knowledge base when the command /index is issued."""
    # Only allow specific users to reindex (could be enhanced with proper admin controls)
    allowed_users = [12345678]  # Replace with actual admin user IDs
    
    if update.effective_user.id not in allowed_users:
        await update.message.reply_text("Sorry, only administrators can reindex the knowledge base.")
        return
    
    await update.message.reply_text("Starting to reindex the knowledge base. This may take some time...")
    
    try:
        global embedding_processor
        # Initialize embedding processor
        if not embedding_processor:
            embedding_processor = EmbeddingProcessor()
        
        # Process documents
        embedding_processor.process_documents("docs")
        
        await update.message.reply_text("✅ Knowledge base reindexed successfully!")
    except Exception as e:
        logger.error(f"Error reindexing knowledge base: {e}")
        await update.message.reply_text(f"❌ Error reindexing knowledge base: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the user message and get a verification response using RAG."""
    message_text = update.message.text
    
    # Send a typing action to indicate the bot is processing
    await update.message.chat.send_action(action="typing")
    
    # Get the consensus answer
    response = await get_consensus_answer(message_text)
    
    # Send the response
    await update.message.reply_text(response)

async def initialize_embedding_processor():
    """Initialize the embedding processor for RAG."""
    global embedding_processor
    
    try:
        logger.info("Initializing embedding processor...")
        embedding_processor = EmbeddingProcessor()
        logger.info("Embedding processor initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing embedding processor: {e}")
        logger.warning("Bot will run without RAG capabilities.")

def main() -> None:
    """Start the bot."""
    if not TOKEN:
        logger.error("Telegram bot token not found. Please set TELEGRAM_BOT_TOKEN environment variable.")
        return
    
    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()

    # Initialize the embedding processor
    asyncio.get_event_loop().run_until_complete(initialize_embedding_processor())

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("index", reindex_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == "__main__":
    main() 