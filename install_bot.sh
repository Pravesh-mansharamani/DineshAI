#!/bin/bash
set -e

echo "Installing CapCheck Telegram Bot..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "# Enter your Telegram bot token here" > .env
    echo "TELEGRAM_BOT_TOKEN=" >> .env
    echo -e "\033[1;33mPlease edit the .env file and add your Telegram bot token\033[0m"
else
    echo ".env file already exists."
fi

echo -e "\n\033[1;32mInstallation completed!\033[0m"
echo -e "\nTo run the bot:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Make sure you've added your Telegram bot token to the .env file"
echo "3. Run: python src/telegram_bot.py"
echo -e "\nDon't have a Telegram bot token? Get one from @BotFather on Telegram." 