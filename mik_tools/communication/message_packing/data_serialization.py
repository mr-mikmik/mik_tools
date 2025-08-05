import numpy as np
import pickle
import json
from enum import Enum
from typing import Optional, Callable, Dict, Any, Tuple


class MessageType(Enum):
    JSON = "json"
    NUMPY = "numpy"
    PICKLE = "pickle"


def serialize_data(data: Any) -> Tuple[bytes, MessageType]:
    """Serialize data based on its type"""
    if isinstance(data, np.ndarray):
        # Handle numpy arrays
        serialized = pickle.dumps({
            'type': 'numpy_array',
            'data': data,
            'dtype': str(data.dtype),
            'shape': data.shape
        })
        return serialized, MessageType.NUMPY

    elif isinstance(data, dict):
        # Check if dictionary contains numpy arrays
        has_numpy = any(isinstance(v, np.ndarray) for v in data.values())
        if has_numpy:
            serialized = pickle.dumps({
                'type': 'numpy_dict',
                'data': data
            })
            return serialized, MessageType.NUMPY
        else:
            # Regular dictionary with basic types
            try:
                serialized = json.dumps(data).encode('utf-8')
                return serialized, MessageType.JSON
            except TypeError:
                # Fallback to pickle for complex objects
                serialized = pickle.dumps(data)
                return serialized, MessageType.PICKLE

    elif isinstance(data, (list, tuple)) and data:
        # Check if list/tuple contains only basic types
        if all(isinstance(item, (int, float, str, bool, type(None))) for item in data):
            serialized = json.dumps(data).encode('utf-8')
            return serialized, MessageType.JSON
        else:
            # Contains complex objects, use pickle
            serialized = pickle.dumps(data)
            return serialized, MessageType.PICKLE

    else:
        # Try JSON first for basic types
        try:
            serialized = json.dumps(data).encode('utf-8')
            return serialized, MessageType.JSON
        except TypeError:
            # Fallback to pickle
            serialized = pickle.dumps(data)
            return serialized, MessageType.PICKLE


def deserialize_data(data: bytes, msg_type: MessageType) -> Any:
    """Deserialize received data"""
    if msg_type == MessageType.JSON:
        return json.loads(data.decode('utf-8'))
    elif msg_type in [MessageType.PICKLE]:
        return pickle.loads(data)
    elif msg_type == MessageType.NUMPY:
        unpacked = pickle.loads(data)
        if unpacked['type'] == 'numpy_array':
            return np.array(unpacked['data'], dtype=unpacked['dtype']).reshape(unpacked['shape'])
        elif unpacked['type'] == 'numpy_dict':
            return {k: np.array(v, dtype=v.dtype) if isinstance(v, np.ndarray) else v for k, v in unpacked['data'].items()}
    else:
        raise ValueError(f"Unsupported message type: {msg_type}")


def double_serialization(data: Any) -> bytes:
    serialized_data, msg_type = serialize_data(data)
    double_serialized_data, double_msg_type = serialize_data({'type': msg_type, 'data': serialized_data})
    assert double_msg_type == MessageType.PICKLE, f"Double serialization failed! -- msg_type: {double_msg_type}"
    return double_serialized_data


def double_deserialization(double_serialized_data:bytes) -> Any:
    """Deserialize data that has been serialized twice"""
    deserialized_data = deserialize_data(double_serialized_data, MessageType.PICKLE)
    data_retreived = deserialize_data(deserialized_data['data'], deserialized_data['type'])
    return data_retreived