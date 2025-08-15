FROM python:3.8-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Use Cloud Run's PORT environment variable (defaults to 8080)
ENV PORT=8080
EXPOSE $PORT

# Run Streamlit on the dynamic port
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false