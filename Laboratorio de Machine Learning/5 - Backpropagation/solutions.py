import numpy as np

"""
Debería resolver esta práctica sin agregar más librerías externas
"""

def NotImplemented_message():
    print('###################################')
    print('Tienen que implementar esta función')
    print('###################################')
    return np.array([1, 1])

def densa_forward(X, W, b):
    return X.dot(W)+b

def MSE(X_true, X_pred): 
    mse = np.square(X_true - X_pred).mean()
    return mse

def MSE_grad(X_true, X_pred):
    return NotImplemented_message()

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_jac(Xin):
    return NotImplemented_message()

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def softmax_jac(Xin):
    return NotImplemented_message()

def forward(X, P_true, weights):
    return NotImplemented_message()

def get_gradients(X, P_true, weights):
    return NotImplemented_message()