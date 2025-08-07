# Use a specific, compatible version of Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Define the command to run your app
CMD ["streamlit", "run", "app.py"]