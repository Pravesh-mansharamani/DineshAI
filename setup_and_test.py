#!/usr/bin/env python
"""
Simplified setup and test script for the Flare Network RAG-Consensus bot.
"""

import os
import sys
import asyncio
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_config():
    """Load the consensus engine configuration."""
    try:
        config_path = "src/flare_ai_consensus/input.json"
        with open(config_path, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load configuration: {e}")
        return None

def check_environment():
    """Check required environment variables."""
    required_vars = ["OPEN_ROUTER_API_KEY", "TELEGRAM_BOT_TOKEN"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Some functionality may be limited")
        return False
    return True

async def test_setup():
    """Test the setup with a simplified approach."""
    logger.info("Testing simplified RAG setup...")
    
    # Check if src directory exists
    if not os.path.exists("src"):
        logger.error("src directory not found")
        return False
    
    # Check if config file exists
    config = load_config()
    if not config:
        logger.error("Cannot proceed without configuration")
        return False
    
    # Create a sample document for testing
    os.makedirs("docs", exist_ok=True)
    if not any(os.path.exists(os.path.join("docs", f)) for f in os.listdir("docs") if os.path.isfile(os.path.join("docs", f))):
        logger.info("Creating a sample document for testing...")
        with open("docs/flare_sample.md", "w") as f:
            f.write("""# Flare Network

Flare is a blockchain that aims to connect everything. It is the blockchain for data.

## Key Features

1. **State Connector** - Enables Flare to access and use data from other blockchains and the internet
2. **FAssets** - Tokens that track the value of other assets
3. **Flare Time Series Oracle (FTSO)** - Provides decentralized price and data feeds

## Native Tokens

- **Flare (FLR)** - The native token used for securing the network
- **Songbird (SGB)** - The canary network token

## Consensus Mechanism

Flare uses the Avalanche consensus protocol with a unique implementation called the Flare Consensus Protocol.
""")
        logger.info("Sample document created")
    
    # Test successful
    logger.info("Basic setup test passed")
    return True

async def main():
    """Main function to run the simplified setup test."""
    logger.info("Starting simplified setup test...")
    
    # Check environment
    env_ok = check_environment()
    if not env_ok:
        logger.warning("Environment check failed, but continuing with limited functionality")
    
    # Run simplified test
    setup_ok = await test_setup()
    
    if setup_ok:
        logger.info("✅ Simplified setup test completed successfully!")
        logger.info("You can now run the Telegram bot with: python src/telegram_bot_consensus_rag.py")
        return 0
    else:
        logger.error("❌ Setup test failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)