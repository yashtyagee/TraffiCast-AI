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

# Command to run Streamlit, configuration is handled via .streamlit/config.toml
CMD ["streamlit", "run", "app.py"]
