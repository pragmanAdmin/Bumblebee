# Python runtime as a parent image
FROM python:3.9-slim

# Working directory in the container
WORKDIR /app

# Copying the current directory contents into the container
COPY . /app

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]
