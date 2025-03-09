#!/bin/bash
set -e

echo "Updating dependencies using uv..."

# Ensure uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Install the package in development mode
echo "Installing package in development mode..."
uv pip install -e .

echo "Dependencies successfully updated!" 