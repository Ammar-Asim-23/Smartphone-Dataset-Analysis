import numpy as np
import pandas as pd
import copy
import math

def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        total_cost += (f_wb - y[i]) ** 2
    total_cost /= (2 * m)
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m-1):
        f_wb = w * x[i] + b
        dj_db += f_wb - y[i]
        dj_dw += (f_wb - y[i]) * x[i]
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    w_history = []
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)
        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
    return w, b, J_history

def prepare_and_run_linear_regression_with_gradient_descent(df, target_column, alpha=0.01, num_iters=1000):
    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' not found in DataFrame columns")
    
    X = df.drop(columns=[target_column]).values.flatten()
    y = df[target_column].values
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch in X and y sizes: X.shape = {X.shape[0]}, y.shape = {y.shape[0]}")
    
    initial_w = 0
    initial_b = 0
    
    w, b, J_history = gradient_descent(X, y, initial_w, initial_b, compute_cost, compute_gradient, alpha, num_iters)
    
    y_pred = w * X + b
    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted'])
    
    return w, b, y_pred_df, J_history

# Example usage:
# Assuming you have a DataFrame `df_train` and the target column is named 'price'
# w, b, y_pred_df, J_history = prepare_and_run_linear_regression_with_gradient_descent(df_train, 'price')
# print("w:", w)
# print("b:", b)
# print("Predicted values DataFrame:")
# print(y_pred_df)
# print("Cost History:", J_history)

# Using the predictions to compute R^2 score
# Assuming you have a DataFrame `df_test` and the target column is named 'price'
