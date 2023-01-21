# Use an official Python runtime as a parent image
FROM rkrikbaev/tf-env:latest

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

LABEL Auth: Krikbayev Rustam
LABEL Email: "rkrikbaev@gmail.com"
ENV REFRESHED_AT 2023-01-12

# Install any needed packages specified in requirements.txt
COPY ./requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
RUN mkdir application
WORKDIR /application

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

COPY ./api.py .
COPY ./model.py .
COPY ./utils.py .

ENV LOG_LEVEL=DEBUG
ENV TIMEOUT=1000

CMD [ "gunicorn", "-b", "0.0.0.0:8005", "api:api", "--timeout", "1000", "--log-level", "debug" ]