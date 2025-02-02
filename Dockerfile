# Use Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run the server
CMD ["gunicorn", "server:app"]
