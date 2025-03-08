import asyncio
import argparse
import structlog
import json
from pathlib import Path

from flare_ai_consensus.consensus import run_consensus
from flare_ai_consensus.router import AsyncOpenRouterProvider
from flare_ai_consensus.settings import settings, Message, ConsensusConfig
from flare_ai_consensus.utils import load_json, save_json

logger = structlog.get_logger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the enhanced Flare AI Consensus with different examples."
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=["science", "medical", "history", "technical", "custom"],
        default="science",
        help="Choose a predefined example or use 'custom' to input your own query.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Custom query to use (only when --example=custom).",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="Custom system prompt to use (only when --example=custom).",
    )
    return parser.parse_args()

async def test_consensus(example_key: str = "example_scientific_inquiry", custom_query: str = "", custom_system: str = "") -> None:
    """Run the consensus process with the specified example key."""
    # Load input configuration
    config_json = load_json(settings.input_path / "input.json")
    settings.load_consensus_config(config_json)
    
    # Load the example prompts
    prompts_json = load_json(settings.input_path / "prompts.json")
    
    # Setup the initial conversation
    if example_key == "custom":
        if not custom_system:
            custom_system = "You are a helpful, knowledgeable assistant. Provide accurate and well-reasoned answers to the user's questions."
        if not custom_query:
            custom_query = "What are the most promising renewable energy technologies for the coming decade?"
        
        initial_conversation = [
            {"role": "system", "content": custom_system},
            {"role": "user", "content": custom_query}
        ]
    else:
        initial_conversation = prompts_json.get(example_key, [])
    
    # Initialize the provider
    provider = AsyncOpenRouterProvider(
        api_key=settings.open_router_api_key,
        base_url=settings.open_router_base_url
    )
    
    logger.info("Starting consensus process", example=example_key)
    # Run the consensus process
    consensus_result = await run_consensus(
        provider,
        settings.consensus_config,
        initial_conversation
    )
    
    # Save the result
    result = {
        "example": example_key,
        "initial_conversation": initial_conversation,
        "consensus_result": consensus_result
    }
    
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = asyncio.get_event_loop().time()
    output_file = output_dir / f"consensus_result_{example_key}_{int(timestamp)}.json"
    save_json(result, output_file)
    
    logger.info("Consensus process completed", output_file=str(output_file))
    
    # Print the result
    print("\n" + "="*80)
    print(f"CONSENSUS RESULT FOR EXAMPLE: {example_key}")
    print("="*80)
    print(consensus_result)
    print("="*80 + "\n")
    
    return consensus_result

async def main() -> None:
    args = parse_arguments()
    
    example_mapping = {
        "science": "example_scientific_inquiry",
        "medical": "example_medical_information",
        "history": "example_historical_analysis",
        "technical": "example_technical_explanation",
        "custom": "custom"
    }
    
    example_key = example_mapping.get(args.example, "example_scientific_inquiry")
    
    await test_consensus(example_key, args.query, args.system_prompt)

if __name__ == "__main__":
    asyncio.run(main()) 