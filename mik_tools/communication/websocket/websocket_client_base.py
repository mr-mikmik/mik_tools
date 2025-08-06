import asyncio
import logging
import websockets
import time
from typing import Optional, Callable, Dict, Any, Tuple
from websockets.sync.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, WebSocketException
from mik_tools.communication.message_packing import serialize_data, deserialize_data, MessageType, double_serialization, double_deserialization


class WebsocketClientBase(object):
    """
    Synchronous WebSocket client base class for connecting to a WebSocket server.
    """
    def __init__(self, host_ip: str = "0.0.0.0", port:int=8000) -> None:
        self._port = port
        self._host_ip = host_ip
        self._uri = f"ws://{host_ip}:{port}"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        self.logger.addHandler(console_handler)
        self._ws, self._server_metadata = None, None
        self._connected = False

    def connect(self):
        """Connect to the WebSocket server."""
        self._wait_for_server()

    def _wait_for_server(self):
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                self._ws = connect(self._uri, compression=None, max_size=None)
                data_received_raw = self._ws.recv()
                self._server_metadata = self.deserialize_data(data_received_raw)
                logging.info(f"Connected to server: {self._server_metadata}")
                return
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def _send_to_server(self, data: Any) -> Any:
        """Send data to the server and receive a response."""
        self.logger.info(f"Sending data: {data}")
        serialized_data = self.serialize_data(data)
        self.logger.debug(f"Sending data (encoded): {serialized_data}")
        self._ws.send(serialized_data)
        response_raw = self._ws.recv()
        self.logger.debug(f"Received raw response: {response_raw}")
        response = self.deserialize_data(response_raw)
        self.logger.info(f"Processed response: {response}")
        return response

    def send_to_server(self, data: Any) -> Any:
        return self._send_to_server(data)
        # return self._connect_and_send_to_server_sync(data)
        # return asyncio.run(self._connect_and_send_to_server(data))

    def _connect_and_send_to_server(self, data: Any) -> Any:
        """Connect to the server and send data synchronously."""
        if not self._connected:
            self._wait_for_server()
            self._connected = True
        return self._send_to_server(data)

    def serialize_data(self, data_to_serialize: Any) -> bytes:
        """Serialize data to bytes."""
        serialized_data = double_serialization(data_to_serialize)
        self.logger.debug(f"Serialized data: {serialized_data}")
        return serialized_data

    def deserialize_data(self, serialized_data: bytes) -> Any:
        """Deserialize bytes back to original data."""
        deserialized_data = double_deserialization(serialized_data)
        self.logger.debug(f"Deserialized data: {deserialized_data}")
        return deserialized_data

    def close(self):
        """Close the WebSocket connection."""
        if self._ws:
            try:
                self._ws.close(code=1000, reason="Client closing connection")
                logging.info("WebSocket connection closed.")
            except WebSocketException as e:
                logging.error(f"Error closing WebSocket: {e}")
            finally:
                self._ws = None
                self._connected = False
        else:
            logging.warning("No WebSocket connection to close.")

