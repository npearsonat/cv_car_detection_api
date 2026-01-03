FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (updated package names for Debian Trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git

# Copy model and code
COPY model_final.pth .
COPY main.py .

# Expose port
EXPOSE 8080

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]