#!/bin/bash
set -e

echo "Setting up and running CapCheck Telegram Bot..."

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

# Check if telegram module is installed, if not install dependencies
if ! python -c "import telegram" &> /dev/null; then
    echo "Installing telegram module and dependencies..."
    pip install python-telegram-bot>=20.8
    pip install python-dotenv==1.0.0
    pip install -e .
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "# Enter your Telegram bot token here" > .env
    echo "TELEGRAM_BOT_TOKEN=" >> .env
    echo -e "\033[1;33mPlease edit the .env file and add your Telegram bot token\033[0m"
    echo "You'll need to restart the bot after adding your token."
    exit 1
fi

# Check if TELEGRAM_BOT_TOKEN is set
if ! grep -q "TELEGRAM_BOT_TOKEN=." .env; then
    echo -e "\033[1;31mTELEGRAM_BOT_TOKEN is not set in .env file.\033[0m"
    echo "Please edit the .env file and add your Telegram bot token."
    echo "Example: TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ"
    exit 1
fi

# Run the bot
echo -e "\033[1;32mStarting bot...\033[0m"
python src/telegram_bot.py 