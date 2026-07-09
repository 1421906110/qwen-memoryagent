FROM python:3.12-slim

WORKDIR /app

# Install system deps (psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY src/ src/
COPY .env ./

# Expose ports
EXPOSE 8000 8001

# Start script (handles both engine and UI)
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
