import os
import json
import logging
from typing import Any

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)
from dotenv import load_dotenv

# Import the consensus engine
from flare_ai_consensus.consensus_engine import run_consensus

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Telegram bot token from environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Load consensus configuration
def load_config() -> dict[str, Any]:
    """Load the consensus engine configuration from input.json."""
    with open("src/flare_ai_consensus/input.json", "r") as file:
        return json.load(file)

# A wrapper function for the consensus engine
async def get_consensus_answer(prompt: str) -> str:
    """
    Get a consensus answer from the flare_ai_consensus module.
    
    Args:
        prompt: The user's message/question to verify
        
    Returns:
        A verified answer from the consensus engine
    """
    config = load_config()
    
    try:
        # Prefix the prompt with a clear instruction for fact verification
        verification_prompt = (
            f"Please verify the following claim or answer the following question with factual information. "
            f"Mark certainty levels explicitly (VERIFIED, LIKELY, UNCERTAIN). "
            f"Claim/Question: {prompt}"
        )
        
        # Run the consensus engine
        result = run_consensus(config, verification_prompt)
        return result
    except Exception as e:
        logger.error(f"Error getting consensus answer: {e}")
        return f"Sorry, I encountered an error while processing your request: {str(e)}"

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    await update.message.reply_text(
        "Welcome to CapCheck - a community notes for Telegram!\n\n"
        "Send me any message, claim, or question, and I'll verify it using multiple AI models to provide a consensus answer.\n\n"
        "Example: 'The Earth is flat.' or 'Did NASA really land on the moon?'"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
    await update.message.reply_text(
        "How to use CapCheck:\n\n"
        "1. Send any claim or question you want to verify\n"
        "2. Wait a moment while multiple AI models analyze it\n"
        "3. Receive a consensus answer with certainty levels marked as VERIFIED, LIKELY, or UNCERTAIN\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the user message and get a verification response."""
    message_text = update.message.text
    
    # Send a typing action to indicate the bot is processing
    await update.message.chat.send_action(action="typing")
    
    # Get the consensus answer
    response = await get_consensus_answer(message_text)
    
    # Send the response
    await update.message.reply_text(response)

def main() -> None:
    """Start the bot."""
    if not TOKEN:
        logger.error("Telegram bot token not found. Please set TELEGRAM_BOT_TOKEN environment variable.")
        return
    
    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == "__main__":
    main() 