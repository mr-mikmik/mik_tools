import numpy as np
from .websocket_client_base import WebsocketClientBase


class WebsocketPolicyClient(WebsocketClientBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, obs:dict) -> np.ndarray:
        action = self.send_to_server(obs)
        return action