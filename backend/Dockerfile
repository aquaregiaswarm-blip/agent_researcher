FROM python:3.11-slim

# Install system dependencies for weasyprint and psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    libpangoft2-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment
ENV DJANGO_SETTINGS_MODULE=backend.settings.prod
ENV PYTHONUNBUFFERED=1

# Collect static files
RUN SECRET_KEY=build-placeholder python manage.py collectstatic --noinput 2>/dev/null || true

# Default port
ENV PORT=8080
EXPOSE ${PORT}

CMD python manage.py migrate --noinput && exec gunicorn backend.wsgi:application --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 1800
