import pickle
import sys
import os
from collections.abc import Mapping, Sequence
import numpy as np

def inspect_pkl(file_path):
    """Inspect a pickle file and print its structure."""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Loaded data from {file_path}")
        print(f"Type: {type(data)}")

        if isinstance(data, Mapping):
            print(f"Keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"  {key}: {type(value)}", end="")
                if hasattr(value, 'shape'):
                    print(f", shape: {value.shape}", end="")
                    if hasattr(value, 'dtype'):
                        print(f", dtype: {value.dtype}")
                    else:
                        print()
                elif isinstance(value, (list, tuple)):
                    print(f", length: {len(value)}")
                    if value and isinstance(value[0], str):
                        print(f"  First element: '{value[0]}'")
                elif isinstance(value, str):
                    print(f", length: {len(value)}")
                else:
                    print()
        elif isinstance(data, (list, tuple)):
            print(f"Length: {len(data)}")
            if data:
                print(f"First element type: {type(data[0])}")
                if isinstance(data[0], str):
                    print(f"First element: '{data[0]}'")
                    if len(data) > 1:
                        print(f"Last element: '{data[-1]}'")
        elif hasattr(data, 'shape'):
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Min: {data.min()}, Max: {data.max()}")
            print(f"Mean: {data.mean():.2f}")
        else:
            print(f"Value: {data}")

    except Exception as e:
        print(f"Error loading {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_pkl.py <pkl_file_path>")
        sys.exit(1)

    inspect_pkl(sys.argv[1])
