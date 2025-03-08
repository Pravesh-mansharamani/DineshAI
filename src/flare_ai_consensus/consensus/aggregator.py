from flare_ai_consensus.router import (
    AsyncOpenRouterProvider,
    ChatRequest,
    OpenRouterProvider,
)
from flare_ai_consensus.settings import AggregatorConfig, Message


def _concatenate_aggregator(responses: dict[str, str]) -> str:
    """
    Aggregate responses by concatenating each model's answer with a structured format
    for easier comparison and analysis.

    :param responses: A dictionary mapping model IDs to their response texts.
    :return: A single aggregated string with clearly marked sections.
    """
    aggregated_text = "# MODEL RESPONSES\n\n"
    
    for model_id, text in responses.items():
        # Clean up the model ID to make it more readable
        model_name = model_id.split('/')[-1].split(':')[0].replace('-', ' ').title()
        
        # Add a clear section header for each model response
        aggregated_text += f"## Model: {model_name}\n\n{text}\n\n"
        aggregated_text += "---\n\n"  # Add separator between responses
    
    return aggregated_text


def centralized_llm_aggregator(
    provider: OpenRouterProvider,
    aggregator_config: AggregatorConfig,
    aggregated_responses: dict[str, str],
) -> str:
    """Use a centralized LLM  to combine responses.

    :param provider: An OpenRouterProvider instance.
    :param aggregator_config: An instance of AggregatorConfig.
    :param aggregated_responses: A string containing aggregated
        responses from individual models.
    :return: The aggregator's combined response.
    """
    # Build the message list.
    messages: list[Message] = []
    messages.extend(aggregator_config.context)

    # Add a system message with the aggregated responses.
    aggregated_str = _concatenate_aggregator(aggregated_responses)
    messages.append(
        {"role": "system", "content": f"Aggregated responses:\n{aggregated_str}"}
    )

    # Add the aggregator prompt
    messages.extend(aggregator_config.prompt)

    payload: ChatRequest = {
        "model": aggregator_config.model.model_id,
        "messages": messages,
        "max_tokens": aggregator_config.model.max_tokens,
        "temperature": aggregator_config.model.temperature,
    }

    # Get aggregated response from the centralized LLM
    response = provider.send_chat_completion(payload)
    return response.get("choices", [])[0].get("message", {}).get("content", "")


async def async_centralized_llm_aggregator(
    provider: AsyncOpenRouterProvider,
    aggregator_config: AggregatorConfig,
    aggregated_responses: dict[str, str],
) -> str:
    """
    Use a centralized LLM (via an async provider) to combine responses.

    :param provider: An asynchronous OpenRouterProvider.
    :param aggregator_config: An instance of AggregatorConfig.
    :param aggregated_responses: A string containing aggregated
        responses from individual models.
    :return: The aggregator's combined response as a string.
    """
    messages = []
    messages.extend(aggregator_config.context)
    messages.append(
        {"role": "system", "content": f"Aggregated responses:\n{aggregated_responses}"}
    )
    messages.extend(aggregator_config.prompt)

    payload: ChatRequest = {
        "model": aggregator_config.model.model_id,
        "messages": messages,
        "max_tokens": aggregator_config.model.max_tokens,
        "temperature": aggregator_config.model.temperature,
    }

    response = await provider.send_chat_completion(payload)
    return response.get("choices", [])[0].get("message", {}).get("content", "")
