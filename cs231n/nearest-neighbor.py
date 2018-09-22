import numpy as np

class NearestNeighbor:
  def __init__(self):
    pass
    
  def train(self, X, Y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    self.Xtr = X
    self.Ytr = Y
    
  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    Ypred = np.zeros(num_test, dtype=self.Ytr.dtype)
    
    for i in arange(num_test):
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
      min_index = np.argmin(distances)
      Ypred[i] = self.Ytr[min_index]
      
      return Ypred
