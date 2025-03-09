#!/bin/bash

# Flare Network RAG-Consensus Bot Startup Script
# This script sets up the environment and starts the Telegram bot

# Set to exit immediately if any command fails
set -e

# Configuration
VENV_DIR="venv"
DOCS_DIR="docs"
ENV_FILE=".env"
REPO_DIR=$(pwd)

# Text formatting
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

# Print banner
echo -e "${BLUE}${BOLD}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║             FLARE NETWORK RAG-CONSENSUS BOT            ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create/check .env file
check_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${YELLOW}Creating .env file...${RESET}"
        touch "$ENV_FILE"
        
        # Ask for OpenRouter API key if not provided
        if [ -z "$OPEN_ROUTER_API_KEY" ]; then
            echo -e "${YELLOW}Enter your OpenRouter API key (required):${RESET}"
            read -r OPEN_ROUTER_API_KEY
        fi
        
        # Ask for Telegram bot token if not provided
        if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
            echo -e "${YELLOW}Enter your Telegram Bot token (required):${RESET}"
            read -r TELEGRAM_BOT_TOKEN
        fi
        
        # Write to .env file
        echo "OPEN_ROUTER_API_KEY=$OPEN_ROUTER_API_KEY" >> "$ENV_FILE"
        echo "TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN" >> "$ENV_FILE"
    else
        echo -e "${GREEN}Found .env file${RESET}"
    fi
    
    # Export environment variables
    export $(grep -v '^#' "$ENV_FILE" | xargs)
}

# Function to check required directories and files
check_required_files() {
    # Check for docs directory
    if [ ! -d "$DOCS_DIR" ]; then
        echo -e "${YELLOW}Creating docs directory...${RESET}"
        mkdir -p "$DOCS_DIR"
        echo -e "${YELLOW}Warning: The docs directory is empty. You should add Flare Network documentation files to it.${RESET}"
    else
        # Count files in docs directory
        FILE_COUNT=$(find "$DOCS_DIR" -type f | wc -l)
        if [ "$FILE_COUNT" -eq 0 ]; then
            echo -e "${YELLOW}Warning: The docs directory is empty. You should add Flare Network documentation files to it.${RESET}"
        else
            echo -e "${GREEN}Found $FILE_COUNT files in docs directory${RESET}"
        fi
    fi
    
    # Check for required directories
    for dir in "src" "src/flare_ai_consensus"; do
        if [ ! -d "$dir" ]; then
            echo -e "${RED}Error: Required directory '$dir' not found${RESET}"
            exit 1
        fi
    done
    
    # Create faiss_index directory if it doesn't exist
    if [ ! -d "faiss_index" ]; then
        echo -e "${YELLOW}Creating faiss_index directory...${RESET}"
        mkdir -p "faiss_index"
    fi
}

# Function to set up virtual environment
setup_venv() {
    # Check if Python is installed
    if ! command_exists python3; then
        echo -e "${RED}Error: Python 3 is not installed. Please install Python 3 and try again.${RESET}"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Creating virtual environment...${RESET}"
        
        # Check if virtualenv is available
        if ! command_exists virtualenv; then
            python3 -m pip install virtualenv
        fi
        
        # Create virtual environment
        python3 -m virtualenv "$VENV_DIR"
        echo -e "${GREEN}Virtual environment created${RESET}"
    else
        echo -e "${GREEN}Found existing virtual environment${RESET}"
    fi
    
    # Activate virtual environment
    echo -e "${YELLOW}Activating virtual environment...${RESET}"
    source "$VENV_DIR/bin/activate"
    
    # Install dependencies
    echo -e "${YELLOW}Installing dependencies...${RESET}"
    # Core dependencies
    pip install langchain langchain-openai faiss-cpu tiktoken python-telegram-bot python-dotenv httpx requests
    # Additional dependencies
    pip install "fastapi>=0.103.1" "python-multipart>=0.0.6" "openai>=0.27.8" "structlog>=23.1.0"
    
    # We're using FAISS instead of Pinecone, so we don't need langchain-pinecone
    echo -e "${BLUE}Note: Using local FAISS vector store instead of Pinecone for document embeddings${RESET}"
    
    echo -e "${GREEN}Dependencies installed${RESET}"
}

# Function to run the setup and test script
return 0
            else
                echo -e "${RED}Setup and test still failing after fixes${RESET}"
                return 1
            fi
        fi
        
        return 1
    fi
}

# Function to start the bot
start_bot() {
    echo -e "${YELLOW}Starting Flare Network RAG-Consensus Bot...${RESET}"
    echo -e "${BLUE}Press Ctrl+C to stop the bot${RESET}"
    
    # Start the bot
    python src/telegram_bot_consensus_rag.py
}

# Main execution
main() {
    echo -e "${YELLOW}Starting setup...${RESET}"
    
    # Check environment file
    check_env_file
    
    # Check required files and directories
    check_required_files
    
    # Setup virtual environment
    setup_venv
    
    # Run setup and test
    if run_setup_and_test; then
        # Start the bot
        start_bot
    else
        echo -e "${RED}Failed to initialize the system. Please check the logs above.${RESET}"
        exit 1
    fi
}

# Run main function
main