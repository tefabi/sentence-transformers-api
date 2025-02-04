# Sentence Transformers API

[![Tests](https://github.com/tefabi/sentence-transformers-api/actions/workflows/tests.yaml/badge.svg)](https://github.com/tefabi/sentence-transformers-api/actions/workflows/tests.yaml) [![Deploy](https://github.com/tefabi/sentence-transformers-api/actions/workflows/deploy.yaml/badge.svg)](https://github.com/tefabi/sentence-transformers-api/actions/workflows/deploy.yaml)  
A REST API for getting embeddings through sentence transformers.

## Installation

You can install and run the api using either of the options below:

1. Docker compose
2. Manual installation

### Docker Compose

Prerequisites:

- You need to have installed docker

Steps:

1. Clone the repository:
   ```shell
   git clone https://github.com/tefabi/sentence-transformers-api
   ```
2. Create a .env and add a token:
   ```shell
   echo "TOKEN=mySecretToken"
   ```
3. Start the service:
   ```shell
   docker compose up -d
   ```

### Manual Installation

Prerequisites:

- You need to have installed python.
  Steps:

1. Clone the repository:
   ```shell
   git clone https://github.com/tefabi/sentence-transformers-api
   ```
2. Create a .env and add a token:
   ```shell
   echo "TOKEN=mySecretToken"
   ```
3. Start a virtual environment, activate it and install the requirements:
   ```shell
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Start the api server
   ```
   fastapi dev
   ```
