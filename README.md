[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Flare](https://img.shields.io/badge/flare-network-e62058.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNCIgaGVpZ2h0PSIzNCI+PHBhdGggZD0iTTkuNC0uMWEzMjAuMzUgMzIwLjM1IDAgMCAwIDIuOTkuMDJoMi4yOGExMTA2LjAxIDExMDYuMDEgMCAwIDEgOS4yMy4wNGMzLjM3IDAgNi43My4wMiAxMC4xLjA0di44N2wuMDEuNDljLS4wNSAyLTEuNDMgMy45LTIuOCA1LjI1YTkuNDMgOS40MyAwIDAgMS02IDIuMDdIMjAuOTJsLTIuMjItLjAxYTQxNjEuNTcgNDE2MS41NyAwIDAgMS04LjkyIDBMMCA4LjY0YTIzNy4zIDIzNy4zIDAgMCAxLS4wMS0xLjUxQy4wMyA1LjI2IDEuMTkgMy41NiAyLjQgMi4yIDQuNDcuMzcgNi43LS4xMiA5LjQxLS4wOXoiIGZpbGw9IiNFNTIwNTgiLz48cGF0aCBkPSJNNy42NSAxMi42NUg5LjJhNzU5LjQ4IDc1OS40OCAwIDAgMSA2LjM3LjAxaDMuMzdsNi42MS4wMWE4LjU0IDguNTQgMCAwIDEtMi40MSA2LjI0Yy0yLjY5IDIuNDktNS42NCAyLjUzLTkuMSAyLjVhNzA3LjQyIDcwNy40MiAwIDAgMC00LjQtLjAzbC0zLjI2LS4wMmMtMi4xMyAwLTQuMjUtLjAyLTYuMzgtLjAzdi0uOTdsLS4wMS0uNTVjLjA1LTIuMSAxLjQyLTMuNzcgMi44Ni01LjE2YTcuNTYgNy41NiAwIDAgMSA0LjgtMnoiIGZpbGw9IiNFNjIwNTciLz48cGF0aCBkPSJNNi4zMSAyNS42OGE0Ljk1IDQuOTUgMCAwIDEgMi4yNSAyLjgzYy4yNiAxLjMuMDcgMi41MS0uNiAzLjY1YTQuODQgNC44NCAwIDAgMS0zLjIgMS45MiA0Ljk4IDQuOTggMCAwIDEtMi45NS0uNjhjLS45NC0uODgtMS43Ni0xLjY3LTEuODUtMy0uMDItMS41OS4wNS0yLjUzIDEuMDgtMy43NyAxLjU1LTEuMyAzLjM0LTEuODIgNS4yNy0uOTV6IiBmaWxsPSIjRTUyMDU3Ii8+PC9zdmc+&colorA=FFFFFF)](https://dev.flare.network/)

# AI Consensus System

A robust framework for generating more accurate, reliable answers by using multiple large language models and an aggregation approach.

## Overview

The AI Consensus System leverages the collective intelligence of multiple AI models to produce more accurate, reliable responses than any single model can achieve. By analyzing responses from different models (Gemini, Claude, and Perplexity), identifying points of agreement and disagreement, and synthesizing a consensus, the system delivers answers with greater factual accuracy and consistency.

## Key Features

- **Multi-Model Analysis**: Combines responses from three leading LLMs (Google's Gemini, Anthropic's Claude, and Perplexity)
- **Rigorous Consensus Building**: Evaluates agreement across models, identifying verified facts vs. uncertain claims
- **Direct, Concise Responses**: Removes meta-commentary and filler language for clear, to-the-point answers
- **Iterative Refinement**: Applies multiple passes of improvement for optimal responses
- **Comprehensive Testing**: Includes a testing framework to measure accuracy, reliability, and performance metrics

## How It Works

1. The system sends the same query to multiple AI models in parallel
2. A specialized aggregator model (Claude) analyzes all responses to identify:
   - Points of strong agreement across models
   - Areas of partial agreement
   - Contradictions or inconsistencies
3. The aggregator synthesizes a consensus response that:
   - Clearly marks information as [VERIFIED], [LIKELY], or [UNCERTAIN]
   - Prioritizes facts with strong model agreement
   - Presents information in a logical, structured format
4. The consensus undergoes iterative refinement to enhance conciseness and clarity
5. The final answer combines the strengths of all models while reducing individual weaknesses

## Testing Framework

The system includes a comprehensive testing suite that evaluates:

### Accuracy Improvements
- **Factual Correctness**: How well the consensus incorporates key facts from ground truth
- **Semantic Similarity**: How closely the meaning matches reference answers
- **Exact Match Performance**: Direct comparison against ground truth

### Reliability Metrics
- **Consistency**: How similar responses are across multiple runs of the same query
- **Uncertainty Handling**: How well the system handles ambiguous or contested topics

### Performance Benchmarks
- **Response Time**: Comparison of consensus system vs. individual models
- **Overhead Assessment**: Additional cost of the consensus approach

## Usage

### Running Tests

```bash
# Run tests with default settings
python -m src.flare_ai_consensus.test_consensus

# Run with custom settings
python -m src.flare_ai_consensus.test_consensus --config=path/to/config.json --test-cases=path/to/test_cases.json --output-dir=results --num-runs=5
```

### Interpreting Results

Test results are saved in two formats:
- A detailed JSON report with all metrics and raw responses
- A human-readable summary text file highlighting key findings

Key metrics to review:
- **Accuracy improvement percentage**: How much better the consensus performs vs. the best single model
- **Reliability improvement**: Consistency gains across test runs
- **Performance overhead**: Additional processing time compared to single models

## Project Structure

```
src/flare_ai_consensus/
├── __init__.py
├── consensus_engine.py    # Core implementation of consensus system
├── input.json             # Configuration for models and prompts
├── test_cases.json        # Test cases with ground truth
└── test_consensus.py      # Testing framework
```

## Benefits

- **Enhanced Accuracy**: By combining multiple models, the system reduces individual model biases and limitations
- **Improved Reliability**: Greater consistency in answers across multiple runs
- **Appropriate Uncertainty**: Clear distinction between verified facts and uncertain claims
- **Concise Responses**: Direct answers without unnecessary explanations or meta-commentary

## Future Improvements

- Integration with semantic similarity metrics for more accurate evaluations
- Support for streaming responses
- Custom weighting of different models based on domain expertise
- Visual representation of model agreement levels

# Flare AI Consensus

Flare AI SDK for Consensus Learning.

## 🚀 Key Features

- **Consensus Learning Implementation**
  A Python implementation of single-node, multi-model Consensus Learning (CL). CL is a decentralized ensemble learning paradigm introduced in [arXiv:2402.16157](https://arxiv.org/abs/2402.16157), which is now being generalized to large language models (LLMs).

- **300+ LLM Support**
  Leverages OpenRouter to access over 300 models via a unified interface.

- **Iterative Feedback Loop**
  Employs an aggregation process where multiple LLM outputs are refined over configurable iterations.

- **Modular & Configurable**
  Easily customize models, conversation prompts, and aggregation parameters through a simple JSON configuration.

## 🎯 Getting Started

Before getting started, ensure you have:

- A **Python 3.12** environment.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed for dependency management.
- [Docker](https://www.docker.com/)
- An [OpenRouter API Key](https://openrouter.ai/settings/keys).

### Build & Run Instructions

You can deploy Flare AI Consensus using Docker or set up the backend and frontend manually.

#### Environment Setup

1. **Prepare the Environment File:**
   Rename `.env.example` to `.env` and update the variables accordingly. (e.g. your [OpenRouter API Key](https://openrouter.ai/keys))

### Build using Docker (Recommended)

1. **Build the Docker Image:**

   ```bash
   docker build -t flare-ai-consensus .
   ```

2. **Run the Docker Container:**

   ```bash
   docker run -p 80:80 -it --env-file .env flare-ai-consensus
   ```

3. **Access the Frontend:**
   Open your browser and navigate to [http://localhost:80/docs](http://localhost:80/docs) to interact with the Chat UI.

### 🛠 Build Manually

Flare AI Consensus is a Python-based backend. Follow these steps for manual setup:

1. **Install Dependencies:**
   Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to install backend dependencies:

   ```bash
   uv sync --all-extras
   ```

   Verify your available credits and get all supported models with:

   ```bash
   uv run python -m tests.credits
   uv run python -m tests.models
   ```

2. **Configure CL instance:**
   Configure your CL instance in `src/input.json`, including:

   - **Models:** Specify each LLM's OpenRouter `id`, along with parameters like `max_tokens` and `temperature`.
   - **Aggregator Settings:** Define the aggregator model, additional context, aggregation prompt, and specify how aggregated responses are handled.
   - **Iterations:** Determine the number of iterations for the feedback loop.

3. **Start the Backend:**
   The backend runs by default on `0.0.0.0:8080`:

   ```bash
   uv run start-backend
   ```

### Testing Endpoints

For granular testing, use the following endpoints:

- **Completion Endpoint (Non-Conversational):**

  ```bash
  uv run python -m tests.completion --prompt "Who is Ash Ketchum?" --model "google/learnlm-1.5-pro-experimental:free"
  ```

- **Chat Completion Endpoint (Conversational):**

  ```bash
  uv run python -m tests.chat_completion --mode default
  ```

  _Tip:_ In interactive mode, type `exit` to quit.

## 📁 Repo Structure

```
src/flare_ai_consensus/
├── attestation/           # TEE attestation implementation
│   ├── simulated_token.txt
│   ├── vtpm_attestation.py
│   └── vtpm_validation.py
├── api/                    # API layer
│   ├── middleware/        # Request/response middleware
│   └── routes/           # API endpoint definitions
├── consensus/             # Core consensus learning
│   ├── aggregator.py      # Response aggregation
│   └── consensus.py       # Main CL implementation
├── router/               # API routing and model access
│   ├── base_router.py     # Base routing interface
│   └── openrouter.py      # OpenRouter implementation
├── utils/                # Utility functions
│   ├── file_utils.py      # File operations
│   └── parser_utils.py    # Input parsing
├── input.json            # Configuration file
├── main.py               # Application entry
└── settings.py           # Environment settings
```

## 🚀 Deploy on TEE

Deploy on a [Confidential Space](https://cloud.google.com/confidential-computing/confidential-space/docs/confidential-space-overview) using AMD SEV.

### Prerequisites

- **Google Cloud Platform Account:**
  Access to the [`verifiable-ai-hackathon`](https://console.cloud.google.com/welcome?project=verifiable-ai-hackathon) project is required.

- **OpenRouter API Key:**
  Ensure your [OpenRouter API key](https://openrouter.ai/settings/keys) is in your `.env`.

- **gcloud CLI:**
  Install and authenticate the [gcloud CLI](https://cloud.google.com/sdk/docs/install).

### Environment Configuration

1. **Set Environment Variables:**
   Update your `.env` file with:

   ```bash
   TEE_IMAGE_REFERENCE=ghcr.io/flare-research/flare-ai-consensus:main  # Replace with your repo build image
   INSTANCE_NAME=<PROJECT_NAME-TEAM_NAME>
   ```

2. **Load Environment Variables:**

   ```bash
   source .env
   ```

   > **Reminder:** Run the above command in every new shell session. On Windows, we recommend using [git BASH](https://gitforwindows.org) to access commands like `source`.

3. **Verify the Setup:**

   ```bash
   echo $TEE_IMAGE_REFERENCE # Expected output: Your repo build image
   ```

### Deploying to Confidential Space

Run the following command:

```bash
gcloud compute instances create $INSTANCE_NAME \
  --project=verifiable-ai-hackathon \
  --zone=us-south1-a \
  --machine-type=n2d-standard-2 \
  --network-interface=network-tier=PREMIUM,nic-type=GVNIC,stack-type=IPV4_ONLY,subnet=default \
  --metadata=tee-image-reference=$TEE_IMAGE_REFERENCE,\
tee-container-log-redirect=true,\
tee-env-OPEN_ROUTER_API_KEY=$OPEN_ROUTER_API_KEY,\
  --maintenance-policy=MIGRATE \
  --provisioning-model=STANDARD \
  --service-account=confidential-sa@verifiable-ai-hackathon.iam.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --min-cpu-platform="AMD Milan" \
  --tags=flare-ai,http-server,https-server \
  --create-disk=auto-delete=yes,\
boot=yes,\
device-name=$INSTANCE_NAME,\
image=projects/confidential-space-images/global/images/confidential-space-debug-250100,\
mode=rw,\
size=11,\
type=pd-standard \
  --shielded-secure-boot \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --reservation-affinity=any \
  --confidential-compute-type=SEV
```

#### Post-deployment

1. After deployment, you should see an output similar to:

   ```plaintext
   NAME          ZONE           MACHINE_TYPE    PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP    STATUS
   consensus-team1   us-central1-b  n2d-standard-2               10.128.0.18  34.41.127.200  RUNNING
   ```

2. It may take a few minutes for Confidential Space to complete startup checks. You can monitor progress via the [GCP Console](https://console.cloud.google.com/welcome?project=verifiable-ai-hackathon) logs.
   Click on **Compute Engine** → **VM Instances** (in the sidebar) → **Select your instance** → **Serial port 1 (console)**.

   When you see a message like:

   ```plaintext
   INFO:     Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
   ```

   the container is ready. Navigate to the external IP of the instance (visible in the **VM Instances** page) to access the docs (`<IP>:80/docs`).

### 🔧 Troubleshooting

If you encounter issues, follow these steps:

1. **Check Logs:**

   ```bash
   gcloud compute instances get-serial-port-output $INSTANCE_NAME --project=verifiable-ai-hackathon
   ```

2. **Verify API Key(s):**
   Ensure that all API Keys are set correctly (e.g. `OPEN_ROUTER_API_KEY`).

3. **Check Firewall Settings:**
   Confirm that your instance is publicly accessible on port `80`.

## 💡 Next Steps

- **Security & TEE Integration:**
  - Ensure execution within a Trusted Execution Environment (TEE) to maintain confidentiality and integrity.
- **Factual Correctness**:
  - In line with the main theme of the hackathon, one important aspect of the outputs generated by the LLMs is their accuracy. In this regard, producing sources/citations with the answers would lead to higher trust in the setup. Sample prompts that can be used for this purpose can be found in the appendices of [arXiv:2305.14627](https://arxiv.org/pdf/2305.14627), or in [James' Coffee Blog](https://jamesg.blog/2023/04/02/llm-prompts-source-attribution).
  - _Note_: only certain models may be suitable for this purpose, as references generated by LLMs are often inaccurate or not even real!
- **Prompt Engineering**:
  - Our approach is very similar to the **Mixture-of-Agents (MoA)** introduced in [arXiv:2406.04692](https://arxiv.org/abs/2406.04692), which uses iterative aggregations of model responses. Ther [github repository](https://github.com/togethercomputer/MoA) does include other examples of prompts that can be used for additional context for the LLMs.
  - New iterations of the consensus learning algorithm could have different prompts for improving the previous responses. In this regard, the _few-shot_ prompting techniques introduced by OpenAI in [arXiv:2005.14165](https://arxiv.org/pdf/2005.14165) work by providing models with a _few_ examples of similar queries and responses in addition to the initial prompt. (See also previous work by [Radford et al.](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).)
  - _Chain of Thought_ prompting techniques are a linear problem solving approach where each step builds upon the previous one. Google's approach in [arXiv:2201.11903](https://arxiv.org/pdf/2201.11903) is to augment each prompt with an additional example and chain of thought for an associated answer. (See the paper for multiple examples.)
- **Dynamic resource allocation and Semantic Filters**:
  - An immediate improvement to the current approach would be to use dynamically-adjusted parameters. Namely, the number of iterations and number of models used in the algorithm could be adjusted to the input prompt: _e.g._ simple prompts do not require too many resources. For this, a centralized model could be used to decide the complexity of the task, prior to sending the prompt to the other LLMs.
  - On a similar note, the number of iterations for making progress could adjusted according to how _different_ are the model responses. Semantic entailment for LLM outputs is an active field of research, but a rather quick solution is to rely on _embeddings_. These are commonly used in RAG pipelines, and could also be used here with _e.g._ cosine similarity. You can get started with [GCloud's text embeddings](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings) -- see [flare-ai-rag](https://github.com/flare-foundation/flare-ai-rag/tree/main) for more details.
  - The use of [LLM-as-a-Judge](https://arxiv.org/pdf/2306.05685) for evaluating other LLM outputs has shown good progress -- see also this [Confident AI blogpost](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method).
  - In line with the previously mentioned LLM-as-a-Judge, a model could potentially be used for filtering _bad_ responses. LLM-Blender, for instance, introduced in [arXiv:2306.02561](https://arxiv.org/abs/2306.02561), uses a PairRanker that achieves a ranking of outputs through pairwise comparisons via a _cross-attention encoder_.
- **AI Agent Swarm**:
  - The structure of the reference CL implementation can be changed to adapt _swarm_-type algorithms, where tasks are broken down and distributed among specialized agents for parallel processing. In this case a centralized LLM would act as an orchestrator for managing distribution of tasks -- see _e.g._ [swarms repo](https://github.com/kyegomez/swarms).

# CapCheck System Overview

## Project Overview
CapCheck is a community notes system built using Consensus AI Voting with no human intervention or bias. The system uses a Flare AI Consensus Learning approach to generate reliable consensus answers from multiple large language models.

## System Architecture

### Core Components

1. **FastAPI Application (`main.py`)**
   - Entry point for the application
   - Sets up API routes and middleware
   - Initializes the consensus configuration

2. **ChatRouter (`api/routes/chat.py`)**
   - Handles chat API endpoints
   - Processes incoming messages through the consensus pipeline
   - Returns aggregated responses

3. **Consensus Engine (`consensus/`)**
   - `consensus.py`: Implements the consensus learning algorithm
   - `aggregator.py`: Provides methods to aggregate responses from different models
   - Runs multiple iterations to refine answers

4. **Model Integration (`router/`)**
   - Connects to OpenRouter API to access different LLM models
   - Handles async communication with models

5. **Settings and Configuration (`settings.py`)**
   - Manages model configurations
   - Stores consensus algorithm parameters
   - Loads configuration from input.json

### Key Functions

1. **`run_consensus()`** - Main function that:
   - Sends the initial prompt to all models
   - Aggregates responses
   - Runs improvement iterations
   - Returns the final consensus answer

2. **`send_round()`** - Sends requests to all models in parallel and collects responses

3. **`async_centralized_llm_aggregator()`** - Aggregates responses from different models into a consensus

4. **Chat API endpoint** - Accepts user messages and returns consensus responses

## RAG Pipeline Implementation Guide

The optimal place to implement a Retrieval-Augmented Generation (RAG) pipeline in the CapCheck system would be between the user input and the consensus process. This can be accomplished by:

### Implementation Locations

1. **Primary Option: API Layer (`api/routes/chat.py`)**
   - In the `chat()` function in `ChatRouter`, before calling `run_consensus()`
   - This would allow augmenting the user message with retrieved context before passing it to models

2. **Secondary Option: Create a RAG Middleware**
   - Create a new module under `flare_ai_consensus/` (e.g., `rag/`)
   - Implement a middleware component that processes messages before they reach the consensus engine

### Recommended Integration Approach

1. Modify the `chat()` function in `ChatRouter` to:
   - Extract key information from the user's message
   - Query a vector database for relevant context
   - Augment the system or user message with this context
   - Pass the enhanced messages to the consensus pipeline

2. This approach allows the existing consensus process to remain unchanged while benefiting from the additional context provided by RAG.

### Technical Requirements for RAG

1. **Vector Database** - To store and retrieve document embeddings
2. **Document Processing Pipeline** - To convert documents into vectors
3. **Query Processing** - To convert user queries into embeddings
4. **Context Integration** - To merge retrieved information with user queries

No code changes are required at this time - this guide provides a roadmap for where and how to implement RAG functionality while preserving the existing consensus learning architecture.
