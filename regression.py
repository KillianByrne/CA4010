import numpy as np 
import pandas as pd 
import warnings 
warnings.filterwarnings("ignore")

class OrdinaryLeastSquares(object):

	def __init__(self):
		self.coefficients = [] 

	def fit(self, X, y):
		if len(X.shape) == 1: X = self._reshape_x(X)
		
		X = self._concatenate_ones(X)
		self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
	
	def predict(self, entry):
		b0 = self.coefficients[0]
		other_betas = self.coefficients[1:]
		prediction = b0
	
		for xi, bi in zip(entry, other_betas): prediction += (bi*xi)
		return prediction

	def _reshape_x(self, X):
		return X.reshape(-1,1)
	def _concatenate_ones(self,X):
		ones = np.ones(shape = X.shape[0]).reshape(-1,1)
		return np.concatenate((ones,X),1)
