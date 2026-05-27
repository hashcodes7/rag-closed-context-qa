#!/bin/bash

# Navigate to project directory (update this path if necessary on EC2)
# cd /path/to/RAG_Closed_context_QA

# Activate virtual environment if you have one
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies if not already installed
pip install -r requirements.txt

# Start Streamlit application in the background
echo "Starting Streamlit on port 8501 at path /askbot..."
nohup streamlit run ui.py > streamlit.log 2>&1 &

echo "Streamlit is running in the background. PID: $!"
echo "Check streamlit.log for output."
