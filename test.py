from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import List
import math
import uvicorn

app = FastAPI()

class MatrixInput(BaseModel):
    matrix: List[List[float]]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Matrix multiplication function (with NumPy)
def matrix_multiply_with_numpy(M, X):
    M = np.array(M)
    X = np.array(X)
    return np.dot(M, X)

# Matrix multiplication function (without NumPy)
def matrix_multiply_without_numpy(M, X):
    result = [[0 for _ in range(len(X[0]))] for _ in range(len(M))]
    for i in range(len(M)):
        for j in range(len(X[0])):
            for k in range(len(X)):
                result[i][j] += M[i][k] * X[k][j]
    return result

# Sigmoid function (element-wise)
def sigmoid(x):
    # If input is a list, apply sigmoid element-wise
    if isinstance(x, list):
        return [[1 / (1 + math.exp(-cell)) for cell in row] for row in x]
    # If input is a numpy array, apply sigmoid element-wise
    return 1 / (1 + np.exp(-x))

# Endpoint: /calculate
@app.post("/calculate")
async def calculate(input_data: MatrixInput):
    # Matrix M (5x5) - example matrix
    M = [[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10],
         [11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20],
         [21, 22, 23, 24, 25]]

    # Matrix X (5x1) from the user's input data
    X = input_data.matrix
    
    # Bias vector B (5x1) - example bias
    B = [1, 1, 1, 1, 1]

    # Perform matrix multiplication and add bias (with NumPy)
    result_with_numpy = matrix_multiply_with_numpy(M, X) + B
    # Apply sigmoid function
    result_with_numpy = sigmoid(result_with_numpy)

    # Perform matrix multiplication and add bias (without NumPy)
    result_without_numpy = matrix_multiply_without_numpy(M, X)
    result_without_numpy = [[result_without_numpy[i][j] + B[i] for j in range(len(result_without_numpy[i]))] for i in range(len(result_without_numpy))] 
    # Apply sigmoid function
    result_without_numpy = sigmoid(result_without_numpy)

    # Return the results
    return {
        "result_with_numpy": result_with_numpy.tolist(),
        "result_without_numpy": result_without_numpy
    }


if __name__ == "__main__":
    uvicorn.run(app)

