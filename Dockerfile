FROM python:3.11-slim

WORKDIR /app

# Copy dependency file first to cache pip layer
COPY requirements.txt .

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Remove config.toml port binding — HF Spaces binds port 7860 itself
# and will conflict if Streamlit also tries to bind it via config
RUN sed -i '/^port/d' .streamlit/config.toml || true

# Hugging Face Spaces requires exposing port 7860
EXPOSE 7860

# Run Streamlit with all HF-required flags passed explicitly
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.enableWebsocketCompression=false", \
     "--browser.gatherUsageStats=false"]
