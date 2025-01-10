from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import List
import math

app = FastAPI()

class MatrixInput(BaseModel):
    matrix: List[List[float]]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))




# use the post decorator directly below this
'''
    Initialize M and B as np arrays
'''
M = np.array([[1, 2, 3, 4, 5],
              [5, 4, 3, 2, 1],
              [2, 2, 2, 2, 2],
              [3, 3, 3, 3, 3],
              [4, 4, 4, 4, 4]])

B = np.array([[1], [1], [1], [1], [1]])


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Formula: MX + B using NumPy
def matrix_operation_numpy(M, X, B):
    return np.dot(M, X) + B

# Formula: MX + B without NumPy
def matrix_operation_manual(M, X, B):
    result = [[sum(a * b for a, b in zip(M_row, X_col)) for X_col in zip(*X)] for M_row in M]
    result = [[result[i][0] + B[i][0]] for i in range(len(result))]
    return result

# Endpoint: /calculate
@app.post("/calculate")
def calculate():
    # Initialize X as a 5x1 matrix
    X = np.array([[1], [2], [3], [4], [5]])

    # Compute using NumPy
    numpy_result = matrix_operation_numpy(M, X, B)
    numpy_sigmoid_result = sigmoid(numpy_result).tolist()


# Compute without NumPy
    manual_result = matrix_operation_manual(M.tolist(), X.tolist(), B.tolist())
    manual_sigmoid_result = [[sigmoid(value[0])] for value in manual_result]

    return {
        "result_with_numpy": numpy_sigmoid_result,
        "result_without_numpy": manual_sigmoid_result
    }


if __name__ == "__main__":
    uvicorn.run(app)

'''
    Create a requirements.txt
    Upload to render
'''

