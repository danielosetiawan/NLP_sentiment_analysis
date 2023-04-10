# Use the official lightweight Python image.
FROM python:3.8

# Set the working directory in the container
ENV APP_HOME /app
ENV PYTHONUNBUFFERED True
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && \
    apt-get -y install gcc

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy application code and all assets into container
COPY --chown=app:app . ./

# Download the model
# RUN python ./sentiment_prediction.py

# Set user for container
# RUN groupadd -r app && \
#    useradd -r -g app app
# USER app

# Run the gunicorn web service on container startup
CMD exec gunicorn \
    --bind 0.0.0.0:8080 \
  #  --log-level info \
  #  --workers 1 \
  #  --threads 8 \
  #  --timeout 0 \
  app:server