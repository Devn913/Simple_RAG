FROM python:3.10-slim

# Install system dependencies including sqlite3
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a data directory that we will mount as a persistent volume
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE="pdf_chat_project.settings"
# Default DATA_DIR (can be overridden by k8s but this is a fallback)
ENV DATA_DIR="/app/data"

# Command to run the application using Gunicorn
CMD ["sh", "-c", "python manage.py migrate && python manage.py collectstatic --noinput && gunicorn pdf_chat_project.wsgi:application --bind 0.0.0.0:8000 --workers 3"]
