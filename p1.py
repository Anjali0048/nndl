import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
y = np.array([0, 0, 0, 1]) 
w1 = 0.2
w2 = 0.4
bias = 0.1 
t = 0.1

def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

for epoch in range(6000):
 for i in range(4):
     ans = x[i][0] * w1 + x[i][1] * w2 + bias 
     result = sigmoid(ans)
     error = y[i] - result 

     w1 += t * error * x[i][0] 
     w2 += t * error * x[i][1] 
     bias += t * error
print("Final outputs after training:")

for i in range(4):
    ans = x[i][0] * w1 + x[i][1] * w2 + bias 
    result = sigmoid(ans)
    print(f"Input: {x[i]}, Output: {result:.4f}, Expected: {y[i]}")