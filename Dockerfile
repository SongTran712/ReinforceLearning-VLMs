# Use the slim Python 3.10 image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your Python file(s) into the container
COPY . .

# Install any dependencies if requirements.txt exists
# (optional - comment out if not needed)
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your Python script (replace script.py with your file name)
CMD ["python", "/app/vlm.py"]
