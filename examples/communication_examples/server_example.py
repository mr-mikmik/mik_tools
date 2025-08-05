import numpy as np
from mik_tools.communication import WebsocketServerBase


class ArrayMultiplyerWebsocketServer(WebsocketServerBase):
    def __init__(self, *args, multiplyer_value=2, **kwargs):
        self.multiplyer_value = multiplyer_value
        super().__init__(*args, **kwargs)

    def _process_data(self, data: dict) -> dict:
        """Process the received data and return a response."""
        processed_data = self._multiply_data(data)
        return processed_data

    def _multiply_data(self, data):
        if isinstance(data, dict):
            data_out = {}
            for key, value in data.items():
                data_out[key] = self._multiply_data(value)
        elif isinstance(data, np.ndarray):
            data_out = data * self.multiplyer_value
        elif type(data) in [int, float]:
            data_out = data * self.multiplyer_value
        elif isinstance(data, list):
            data_out = [self._multiply_data(item) for item in data]
        elif isinstance(data, tuple):
            data_out = tuple(self._multiply_data(item) for item in data)
        else:
            data_out = data
        return data_out


if __name__ == "__main__":
    server = ArrayMultiplyerWebsocketServer(host_ip='localhost', port=8765, multiplyer_value=3)
    server.serve_forever()