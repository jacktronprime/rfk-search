# Use an official Python runtime as the base image
FROM python:3.11-slim

# Install Tesseract and other system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Render will use (Render sets the PORT environment variable)
EXPOSE $PORT

# Define the command to run your app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
