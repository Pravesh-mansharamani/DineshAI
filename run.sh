#!/bin/bash

# Simple run script for Flare Network RAG-Consensus Bot
set -e

# Format options
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

echo -e "${GREEN}=== Starting Flare Network RAG-Consensus Bot ===${RESET}"

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is required but not installed. Please install Python 3.${RESET}"
    exit 1
fi

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${RESET}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${RESET}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing required packages...${RESET}"
pip install langchain langchain-openai faiss-cpu tiktoken python-telegram-bot python-dotenv httpx requests
pip install fastapi python-multipart openai structlog numpy

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file...${RESET}"
    echo -e "${YELLOW}Enter your OpenRouter API key:${RESET}"
    read -r OPEN_ROUTER_API_KEY
    echo -e "${YELLOW}Enter your Telegram Bot token:${RESET}"
    read -r TELEGRAM_BOT_TOKEN
    
    echo "OPEN_ROUTER_API_KEY=$OPEN_ROUTER_API_KEY" > .env
    echo "TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN" >> .env
fi

# Run setup test
echo -e "${YELLOW}Running setup test...${RESET}"
python setup_and_test.py

# Start the bot
echo -e "${GREEN}Starting the bot...${RESET}"
python src/telegram_bot.py