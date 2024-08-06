# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the default command to run the gridSearchCV.py script
# And then start the Flask app
CMD ["sh", "-c", "python gridSearchCV.py && flask run --host=0.0.0.0 --port=5000"]
