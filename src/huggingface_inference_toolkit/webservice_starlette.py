import os
from pathlib import Path
from time import perf_counter

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.endpoints import WebSocketEndpoint
from starlette.routing import WebSocketRoute

from huggingface_inference_toolkit.logging import logger
import asyncio

import sys
sys.path.append('speech-to-speech')
from s2s_handler import EndpointHandler

async def prepare_handler():
    global inference_handler
    inference_handler = EndpointHandler()
    inference_handler.pipeline_manager.start()
    logger.info("Model initialized successfully")

async def health(request):
    return PlainTextResponse("Ok")

class WebSocketPredictEndpoint(WebSocketEndpoint):
    encoding = "bytes"

    async def on_connect(self, websocket):
        await websocket.accept()

    async def on_receive(self, websocket, data):
        # Run the handler's processing in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        response_data = await loop.run_in_executor(None, inference_handler.process_streaming_data, data)
        if response_data:
            await websocket.send_bytes(response_data)

    async def on_disconnect(self, websocket, close_code):
        pass

app = Starlette(
    debug=False,
    routes=[
        Route("/", health, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        WebSocketRoute("/ws", WebSocketPredictEndpoint)
    ],
    on_startup=[prepare_handler],
)
