#!/bin/bash

# Quick Start Script for IrisTrading RL Training

set -e

echo "========================================="
echo "IrisTrading RL Training Pipeline Setup"
echo "========================================="
echo ""

# Check if Python venv exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
else
    echo "Python virtual environment already exists."
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --quiet torch numpy

# Build Rust binary
echo "Building Rust binary (release mode)..."
cargo build --release

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To run training:"
echo "  source .venv/bin/activate"
echo "  VIRTUAL_ENV=\$PWD/.venv cargo run --release -- --mode train"
echo ""
echo "To customize training, edit config.toml"
echo ""
