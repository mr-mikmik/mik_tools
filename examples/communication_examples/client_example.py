from mik_tools.communication import WebsocketClientBase
import time
import numpy as np


class WebsocketClientTest(WebsocketClientBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    client = WebsocketClientTest(host_ip='localhost', port=8765)
    client.connect()
    query = {'array': [1, 2, 3], 'value': 10}
    response = client.send_to_server(query)
    print("Response from server:", response)
    query_2 = {'ar_1': np.array([1, 2.5, 3.5]), 'ar_2': np.array([4, 5, 6.0])}
    response_2 = client.send_to_server(query_2)
    print("Response from server for query 2:", response_2)
    # now a random array case
    query_3 = {'random_array': np.random.rand(5, 5)}
    response_3 = client.send_to_server(query_3)
    print("Response from server for query 3:", response_3)
client.close()