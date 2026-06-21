#!/usr/bin/env python3
"""
Startup wrapper for Hugging Face Spaces.

Problem: Streamlit takes ~50s to import heavy ML libraries and load models.
During that time port 7860 is closed, so HF's health checker never sees a
response and the Space stays stuck on "Starting...".

Solution: This script immediately starts a tiny HTTP server on port 7860 that
returns 200 OK to any request. Once Streamlit is ready and binds to 7860,
this placeholder server shuts down automatically (bind conflict) and
Streamlit takes over.
"""
import subprocess, threading, time, sys, os
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 7860
PLACEHOLDER_HTML = b"""<!DOCTYPE html>
<html><head><title>TraffiCast AI</title>
<style>
body{margin:0;display:flex;align-items:center;justify-content:center;
height:100vh;background:#0f172a;color:#94a3b8;font-family:sans-serif;}
.loader{text-align:center}
.spinner{width:40px;height:40px;border:4px solid #334155;
border-top-color:#3b82f6;border-radius:50%;animation:spin 1s linear infinite;
margin:0 auto 16px}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head><body><div class="loader">
<div class="spinner"></div>
<p>Loading TraffiCast AI &mdash; please wait&hellip;</p>
</div></body></html>"""

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.write = self.wfile.write
        self.write(PLACEHOLDER_HTML)
    def log_message(self, fmt, *args):
        pass  # silence logs

def run_placeholder():
    """Run placeholder HTTP server until Streamlit takes over the port."""
    try:
        srv = HTTPServer(("0.0.0.0", PORT), HealthHandler)
        print(f"[startup] Placeholder health-check server listening on :{PORT}")
        srv.serve_forever()
    except OSError:
        # Port already taken by Streamlit – expected, just exit
        print("[startup] Port taken by Streamlit, placeholder exiting.")

def main():
    # 1. Start placeholder server in a background thread
    t = threading.Thread(target=run_placeholder, daemon=True)
    t.start()

    # 2. Launch Streamlit as a subprocess
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=7860",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.enableWebsocketCompression=false",
        "--browser.gatherUsageStats=false",
    ]
    print(f"[startup] Launching Streamlit: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd="/app")

    # 3. Wait for Streamlit to exit (should run forever)
    sys.exit(proc.wait())

if __name__ == "__main__":
    main()
