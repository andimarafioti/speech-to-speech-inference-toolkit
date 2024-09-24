import os
from pathlib import Path
from time import perf_counter

import orjson
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route
from starlette.endpoints import WebSocketEndpoint
from starlette.websockets import WebSocket
from starlette.routing import WebSocketRoute

from huggingface_inference_toolkit.async_utils import async_handler_call
from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.serialization.base import ContentType
from huggingface_inference_toolkit.serialization.json_utils import Jsoner
from huggingface_inference_toolkit.utils import (
    convert_params_to_int_or_bool,
)
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


async def predict(request):
    try:
        # extracts content from request
        content_type = request.headers.get("content-Type", None)
        # try to deserialize payload
        deserialized_body = ContentType.get_deserializer(content_type).deserialize(
            await request.body()
        )
        # checks if input schema is correct
        if "inputs" not in deserialized_body and "instances" not in deserialized_body:
            raise ValueError(
                f"Body needs to provide a inputs key, received: {orjson.dumps(deserialized_body)}"
            )

        # check for query parameter and add them to the body
        if request.query_params and "parameters" not in deserialized_body:
            deserialized_body["parameters"] = convert_params_to_int_or_bool(
                dict(request.query_params)
            )

        # tracks request time
        start_time = perf_counter()
        # run async not blocking call
        pred = await async_handler_call(inference_handler, deserialized_body)
        # log request time
        logger.info(
            f"POST {request.url.path} | Duration: {(perf_counter()-start_time) *1000:.2f} ms"
        )

        # response extracts content from request
        accept = request.headers.get("accept", None)
        if accept is None or accept == "*/*":
            accept = "application/json"
        # deserialized and resonds with json
        serialized_response_body = ContentType.get_serializer(accept).serialize(
            pred, accept
        )
        return Response(serialized_response_body, media_type=accept)
    except Exception as e:
        logger.error(e)
        return Response(
            Jsoner.serialize({"error": str(e)}),
            status_code=400,
            media_type="application/json",
        )


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
    debug=True,
    routes=[
        Route("/", health, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/", predict, methods=["POST"]),
        Route("/predict", predict, methods=["POST"]),
        WebSocketRoute("/ws", WebSocketPredictEndpoint)
    ],
    on_startup=[prepare_handler],
)
