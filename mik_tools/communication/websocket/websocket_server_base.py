import logging
import traceback
import websockets
import asyncio
import json
import numpy as np
import pickle
import gzip
import abc
from typing import Optional, Any

from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed, WebSocketException

from mik_tools.communication.message_packing import (
    serialize_data,
    deserialize_data,
    double_serialization,
    double_deserialization,
)


class WebsocketServerBase(object):
    """
    Synchronous WebSocket server base class for handling WebSocket connections.
    """

    def __init__(self, host_ip: str = "0.0.0.0", port: int = 8000, metadata: Optional[dict] = None) -> None:
        self._host_ip = host_ip
        self._port = port
        self._uri = f"ws://{host_ip}:{port}"
        self._metadata = metadata or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def serve_forever(self) -> None:
        self.logger.info(f"Server running on {self._uri}")
        server = serve(self._handler, self._host_ip, self._port)
        server.serve_forever()
        print('HH')

    def _handler(self, websocket) -> None:
        """Handle incoming websocket connections synchronously."""
        print('hello')
        self.logger.info(f'Connection from {websocket.remote_address} opened')
        try:
            websocket.send(self.serialize_data(self._metadata))
            self.logger.info("Client connected.")
            while True:
                try:
                    received_data_raw = websocket.recv()
                    received_data = self.deserialize_data(received_data_raw)
                    self.logger.info(f"Raw received: {received_data_raw}")
                    response_data = self._process_data(received_data)
                    self.logger.debug(f"Processed response: {response_data}")
                    response = self.serialize_data(response_data)
                    websocket.send(response)
                except ConnectionClosed:
                    self.logger.info(f"Connection from {websocket.remote_address} closed")
                    break
                except Exception:
                    tb = traceback.format_exc()
                    websocket.send(tb.encode('utf-8'))
                    websocket.close(code=1011, reason="Internal server error. Traceback sent.")
                    raise
        finally:
            self.logger.info(f"Connection from {websocket.remote_address} cleaned up.")

    @abc.abstractmethod
    def _process_data(self, data: Any) -> Any:
        """Process the received data and return a response."""
        pass

    def serialize_data(self, data_to_serialize: Any) -> bytes:
        serialized_data = double_serialization(data_to_serialize)
        self.logger.debug(f"Serialized data: {serialized_data}")
        return serialized_data

    def deserialize_data(self, serialized_data: bytes) -> Any:
        deserialized_data = double_deserialization(serialized_data)
        self.logger.debug(f"Deserialized data: {deserialized_data}")
        return deserialized_data


class AsynchroWebsocketServerBase(object):

    def __init__(self, host_ip:str="0.0.0.0", port:int=8000, metadata:Optional[dict]=None) -> None:
        self._port = port
        self._host_ip = host_ip
        self._uri = f"ws://{host_ip}:{port}"
        self.clients = set()
        self._metadata = metadata or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        self.logger.info(f"Server running on {self._uri}")
        async with websockets.serve(self._handler, self._host_ip, self._port) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        """Handle incoming websocket connections."""
        self.logger.info(f'Connection from {websocket.remote_address} opened')
        await websocket.send(self.serialize_data(self._metadata))
        self.logger.info("Client connected.")
        while True:
            try:
                received_data_raw = await websocket.recv()
                received_data = self.deserialize_data(received_data_raw)
                self.logger.info(f"Raw received: {received_data_raw}")
                response_data = self._process_data(received_data)
                self.logger.debug(f"Processed response: {response_data}")
                response = self.serialize_data(response_data)
                await websocket.send(response)
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
            finally:
                self.clients.remove(websocket)
                return
    @abc.abstractmethod
    def _process_data(self, data: Any) -> Any:
        """Process the received data and return a response."""
        pass

    def serialize_data(self, data_to_serialize:Any) -> bytes:
        """Serialize data to bytes."""
        serialized_data = double_serialization(data_to_serialize)
        self.logger.debug(f"Serialized data: {serialized_data}")
        return serialized_data

    def deserialize_data(self, serialized_data: bytes) -> Any:
        """Deserialize bytes back to original data."""
        deserialized_data = double_deserialization(serialized_data)
        self.logger.debug(f"Deserialized data: {deserialized_data}")
        return deserialized_data


