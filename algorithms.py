import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

class CustomDecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])
    
    def _build_tree(self, X, y, depth):
    
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)
   
        feature_idx = np.argmax(np.var(X, axis=0))  
        split_value = np.mean(X[:, feature_idx])
        
   
        left_mask = X[:, feature_idx] < split_value
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        
   
        left_tree = self._build_tree(X_left, y_left, depth + 1)
        right_tree = self._build_tree(X_right, y_right, depth + 1)
        
        return {'feature_idx': feature_idx,
                'split_value': split_value,
                'left_tree': left_tree,
                'right_tree': right_tree}
    
    def _predict_tree(self, x, tree):
        if isinstance(tree, (float, np.float64)):  
            return tree
        if x[tree['feature_idx']] < tree['split_value']:
            return self._predict_tree(x, tree['left_tree'])
        else:
            return self._predict_tree(x, tree['right_tree'])

class CustomRandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []
        
    def fit(self, X, y):
        self.estimators = []
        X_np = X.values if isinstance(X, pd.DataFrame) else X  
        y_np = y.values if isinstance(y, pd.Series) else y  
        
        for _ in range(self.n_estimators):
            idx = np.random.choice(len(X_np), size=len(X_np), replace=True)
            X_bootstrapped = X_np[idx]
            y_bootstrapped = y_np[idx]
            
            tree = CustomDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_bootstrapped, y_bootstrapped)
            self.estimators.append(tree)
    
    def predict(self, X):
        X_np = X.values if isinstance(X, pd.DataFrame) else X 
        predictions = np.zeros(len(X_np))
        for tree in self.estimators:
            predictions += tree.predict(X_np)
        return predictions / self.n_estimators

def apply_custom_random_forest(X, y):
    rf = RandomForestRegressor()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CustomRandomForestRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error : {mse}\nMean Absolute Error : {mae}\nR2 Score : {r2}')

    with open('models/custom_random_forest.pb','wb') as hd:
        pickle.dump(model, hd)

    return y_pred    


class CustomLinearRegression:
    
    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept
    
    def __repr__(self):
        return "I am a Linear Regression model!"
    
    def fit(self, X, y):
        """
        Fit model coefficients.

        Arguments:
        X: 1D or 2D numpy array 
        y: 1D numpy array
        """
        
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # features and data
        self.features_ = X
        self.target_ = y
        
        # degrees of freedom of population dependent variable variance
        self.dft_ = X.shape[0] - 1   
        # degrees of freedom of population error variance
        self.dfe_ = X.shape[0] - X.shape[1] - 1
            
        # add bias if fit_intercept is True
        if self._fit_intercept:
            X_biased = np.c_[np.ones(X.shape[0]), X]
        else:
            X_biased = X
        
        # closed form solution
        xTx = np.dot(X_biased.T, X_biased)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X_biased.T, y)
        coef = np.dot(inverse_xTx, xTy)
        
        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef
            
        # Predicted/fitted y
        self.fitted_ = np.dot(X,self.coef_) + self.intercept_
        
        # Residuals
        residuals = self.target_ - self.fitted_
        self.resid_ = residuals
    
    def predict(self, X):
        """Output model prediction.

        Arguments:
        X: 1D or 2D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        self.predicted_ = self.intercept_ + np.dot(X, self.coef_)
        return self.predicted_      

def use_custom_linear_regression(X_train,y_train,X_test):
    lr = CustomLinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    with open('models/custom_linear_regression.pb','wb') as hd:
        pickle.dump(lr, hd)
    return y_pred        