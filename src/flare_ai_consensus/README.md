# Enhanced Flare AI Consensus System

This folder contains an improved version of the Flare AI Consensus Learning system. The enhancements focus on creating more rigorous, accurate, and transparent consensus results from multiple large language models.

## Key Improvements

### 1. Model Selection
We've upgraded the system to use the latest and most capable models available through OpenRouter:

- **Llama 3.3 70B**: Meta's most advanced open-source LLM with excellent reasoning capabilities
- **Gemini 2.0 Pro**: Google's powerful model with strong analytical abilities
- **Gemini 2.0 Flash Thinking**: Specialized for generating detailed reasoning chains
- **Mistral 7B**: A compact but powerful model from Mistral AI
- **Qwen 2.5 VL 72B**: Advanced multimodal model from Qwen

These models represent diverse AI architectures and training approaches, ensuring broader knowledge coverage and reducing shared biases.

### 2. Enhanced Prompting System
The prompting system has been redesigned to:

- Request explicit marking of confidence levels (VERIFIED, LIKELY, UNCERTAIN)
- Require models to analyze agreement levels across responses
- Produce a limitations section to highlight uncertainty areas
- Structure responses more clearly with improved formatting
- Include iteration-specific guidance in improvement rounds

### 3. Rigorous Consensus Process
The consensus process now:

- Uses a more structured approach to format model responses for better comparison
- Implements a 3-iteration improvement cycle (increased from 2)
- Provides clearer context between iterations about previous responses
- Uses a lower temperature (0.3) for the aggregator model to increase consistency
- Allows longer responses (increased max_tokens to 1000) for more comprehensive analyses

### 4. Domain-Specific Example Conversations
Added new example conversations in the prompts.json file for:

- Scientific inquiries
- Medical information
- Historical analysis
- Technical explanations

These serve as high-quality templates for different types of consensus questions.

## How to Use

The system works exactly as before, but now produces more nuanced and reliable consensus answers with:

1. Clear marking of confidence levels
2. Better identification of agreements and disagreements
3. More comprehensive explanations
4. Explicit limitations sections

When receiving a response, look for the VERIFIED, LIKELY, and UNCERTAIN markers to understand confidence levels in different parts of the response. 