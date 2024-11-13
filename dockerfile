# Use the official Python image as a base
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install pipenv
RUN pip install pipenv

# Install dependencies from Pipfile.lock
RUN pipenv install --deploy

# Copy the data folder and necessary Python scripts
COPY data/ ./data
COPY predict.py train.py ./

# Expose the port that FastAPI will be using
EXPOSE 8000

# Run the train.py script followed by predict.py script
CMD ["sh", "-c", "pipenv run python train.py && pipenv run python predict.py"]
