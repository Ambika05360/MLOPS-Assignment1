# Use a base image with Python
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY gridSearchCV.py .
COPY app.py .

# Expose the port Flask is running on (if needed for Flask)
EXPOSE 5000

# Define the default command to run the gridSearchCV.py script
CMD ["python", "gridSearchCV.py"]

