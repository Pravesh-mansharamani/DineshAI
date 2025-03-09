#!/bin/bash
set -e

echo "Starting Flare Blockchain Telegram Bot..."

# Ensure we're in the project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import telegram" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check for required environment variables
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with the following variables:"
    echo "TELEGRAM_BOT_TOKEN=your_telegram_bot_token"
    echo "OPEN_ROUTER_API_KEY=your_openai_api_key"
    echo "PINECONE_API_KEY=your_pinecone_api_key"
    echo "PINECONE_INDEX=your_pinecone_index_name"
    exit 1
fi

# Run the bot
echo "Starting the bot..."
python src/flare_telegram_bot.py 