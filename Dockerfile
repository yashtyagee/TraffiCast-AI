FROM python:3.11-slim

WORKDIR /app

# Copy dependency file first to cache pip layer
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Hugging Face Spaces requires exposing port 7860
EXPOSE 7860

# Command to run Streamlit in headless mode to prevent email prompts and telemetry collection hangs
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
