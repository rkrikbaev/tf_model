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

# log folder
RUN mkdir -p /logs

# Copy the current directory contents into the container at /app
RUN mkdir app
WORKDIR /app


# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser
RUN chown -R appuser /app
RUN chown -R appuser /logs

USER appuser

COPY app/ .

ENV LOG_LEVEL=DEBUG
ENV TIMEOUT=1000

CMD [ "gunicorn", "-b", "0.0.0.0:8005", "api:api", "--timeout", "1000", "--log-level", "debug" ]
