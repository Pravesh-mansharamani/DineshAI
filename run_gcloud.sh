#!/bin/bash

# Path to your Google Cloud SDK
GCLOUD_SDK_PATH="/Users/pravesh/Downloads/google-cloud-sdk 2"

# Use Python 3.11 to run gcloud
CLOUDSDK_PYTHON=/usr/local/bin/python3.11 "$GCLOUD_SDK_PATH/bin/gcloud" "$@" 