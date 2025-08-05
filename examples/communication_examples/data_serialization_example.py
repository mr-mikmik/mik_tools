import numpy as np
import pickle
from mik_tools.communication.message_packing import serialize_data, deserialize_data, MessageType


def test_serialization():
    # Example data to serialize
    data = {
        'message': 'Hello, World!',
        'number': 42,
        'array': [1, 2, 3, 4, 5],
        'mik': 5.0,
        'nested_dict': {'key1': 'value1', 'key2': 'value2', 'key3': 2, 'key4': [1, 2, 3]},
    }

    data_2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    # array dictionary case
    data_3 = {
        'ar_1': np.array([1, 2, 3], dtype=np.float32),
        'ar_2': np.array([[1, 2], [3, 4]], dtype=np.float32),
        'ar_3': np.array([[1, 2], [3, 4]], dtype=np.float64),
        'zeros': np.zeros((2, 3), dtype=np.float32),
        'ones': np.ones((2, 3), dtype=np.float32),
    }

    # Evaluate serialization and deserialization
    evaluate_serialization(data)
    print("-----------------------------------")
    evaluate_serialization(data_2)
    print("-----------------------------------")
    evaluate_serialization(data_3)

def evaluate_serialization(data_to_serialize):
    # Serialize the data
    serialized_data, msg_type = serialize_data(data_to_serialize)
    print(f"Original Data: {data_to_serialize}")
    print(f"Message Type: {msg_type}")
    print(f"Serialized Data: {serialized_data}")

    # Deserialize the data
    deserialized_data = deserialize_data(serialized_data, msg_type)
    print(f"Deserialized Data: {deserialized_data}")

    # assert data_to_serialize == deserialized_data, "Deserialized data does not match original data!"


# evaluate double serialization
def evaluate_dobuble_serialization():
    # we can ecnode the type and serialized data again in a dictionary
    data_ar = {
        'ar_1': np.array([1, 2, 3], dtype=np.float32),
        'ar_2': np.array([[1, 2], [3, 4]], dtype=np.float32),
        'ar_3': np.array([[1, 2], [3, 4]], dtype=np.float64),
        'zeros': np.zeros((2, 3), dtype=np.float32),
        'ones': np.ones((2, 3), dtype=np.float32),
    }
    serialized_data, msg_type = serialize_data(data_ar)
    double_serialized_data, msg_type = serialize_data({'type': msg_type, 'data': serialized_data})
    assert msg_type == MessageType.PICKLE, f"Double serialization failed! -- msg_type: {msg_type}"
    print(f"Double Serialized Data: {double_serialized_data}")
    deserialized_data = deserialize_data(double_serialized_data, MessageType.PICKLE)
    data_retreived = deserialize_data(deserialized_data['data'], deserialized_data['type'])
    print(f"Deserialized Double Data: {deserialized_data}")
    print(f"Data Retrieved: {data_retreived}")




if __name__ == '__main__':
    # test_serialization()
    evaluate_dobuble_serialization()
    print("Serialization and deserialization test passed successfully!")