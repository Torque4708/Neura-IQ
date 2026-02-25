#!/bin/bash

source /home/anand/Neura-IQ/env/bin/activate

if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    nohup ollama serve > /home/anand/Neura-IQ/ollama.log 2>&1 &
    sleep 5
fi

cd /home/anand/Neura-IQ

echo "Launching Neura-IQ Multimodal AI Research Assistant..."
streamlit run core.py --server.port 8501 --server.headless true


