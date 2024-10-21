#!/bin/bash

# Define the default port
PORT=80  # 80 is the default port for inference endpoints

# Check if AIP_MODE is set and adjust the port for Vertex AI
if [[ ! -z "${AIP_MODE}" ]]; then
  PORT=${AIP_HTTP_PORT}
fi

# Start the server
uvicorn webservice_starlette:app --host 0.0.0.0 --port ${PORT}
