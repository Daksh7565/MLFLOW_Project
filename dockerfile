# 1. Base Image: Use Python 3.10 slim (lightweight Linux)
FROM python:3.11

# 2. System Dependencies
# Install the system libraries required by OpenCV (GL and GLib)
RUN apt-get update && apt-get install -y \
    libgl1 \           
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# 3. Work Directory
WORKDIR /app

# 4. Install Dependencies
# Copy requirements first to leverage caching
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# 5. Copy Source Code
# This copies main.py, pipelines/, steps/, etc.
COPY . .

# 6. Default Command
# Run the pipeline when the container starts
CMD ["python", "main.py"]