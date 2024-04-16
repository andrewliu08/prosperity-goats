import numpy as np
import pandas as pd

data = {
    "Multiplier": [
        24, 70, 41, 21, 60,
        47, 82, 87, 80, 35,
        73, 89, 100, 90, 17,
        77, 83, 85, 79, 55,
        12, 27, 52, 15, 30
    ],
    "Hunters": [
        2, 4, 3, 2, 4,
        3, 5, 5, 5, 3,
        4, 5, 8, 7, 2,
        5, 5, 5, 5, 4,
        2, 3, 4, 2, 3
    ]
}


matrix = np.zeros((25, 25))
b = np.zeros(25)
# m_1 / (v_1+h_1) = m_2 / (v_2+h_2)
# m_2 * (v_1+h_1) = m_1 * (v_2+h_2)
# m_2*v_1 + m_2*h_1 = m_1*v_2 + m_1*h_2
# m_2*v_1 - m_1*v_2 = m_1*h_2 - m_2*h_1
for i in range(0, 24):
    matrix[i][i] = data["Multiplier"][i + 1]
    matrix[i][i + 1] = -data["Multiplier"][i]
    b[i] = (
        data["Multiplier"][i] * data["Hunters"][i + 1]
        - data["Multiplier"][i + 1] * data["Hunters"][i]
    )

for i in range(0, 25):
    matrix[24][i] = 1
b[24] = 1

x = np.linalg.solve(matrix, b)

sum=0
for i in range(25):
    print(data["Multiplier"][i] * 7500 / (data["Hunters"][i] + x[i]))
    if(x[i]<0):
        sum+=x[i]
print(sum)
print("Solution:", x)
