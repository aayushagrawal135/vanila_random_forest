from imports import *
from dec_tree import *

class TreeEnsemble:
    
    def __init__(self, X, Y, n_trees, sample_sz, min_leaf):
        np.random.seed(42)
        self.X = X
        self.Y = Y
        self.n_trees = n_trees
        self.sample_sz = sample_sz
        self.min_leaf = min_leaf

        self.trees = [self.create_tree() for i in range(n_trees)]
    
    def create_tree(self):
        rnd_ids = np.random.permutation(len(self.Y))[:self.sample_sz]
        x = self.X.iloc[rnd_ids]
        y = self.Y[:self.sample_sz]
        return DecisionTree(x, y, rnd_ids, self.min_leaf)
    
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis = 0)