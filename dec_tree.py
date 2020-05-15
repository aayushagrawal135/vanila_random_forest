from imports import *

class DecisionTree:
    
    def __init__(self, x, y, idxs, min_leaf = 1):
        self.x = x
        self.y = y
        self.min_leaf = min_leaf
        self.n = len(y)
        self.c = x.shape[1]
        self.val = np.mean(y)
        self.score = float('inf')
        self.idxs = idxs
        self.var_idx = None
        
        self.varsplit()
        
    def varsplit(self):
        for i in range(self.c):
            self.find_better_split(i)
            
        if self.is_leaf:
            return
        
        lhs_ids = self.x.index[self.x[self.split_name] <= self.split]
        rhs_ids = self.x.index[self.x[self.split_name] > self.split]
        
        lhs_ids = np.array(lhs_ids)
        rhs_ids = np.array(rhs_ids)

        left_rows = self.x.index.isin(lhs_ids)

        self.lhs = DecisionTree(self.x[left_rows], self.y[left_rows], lhs_ids, self.min_leaf)
        self.rhs = DecisionTree(self.x[~left_rows], self.y[~left_rows], rhs_ids, self.min_leaf)

    def find_better_split(self, var_idx):
        x = self.x.values[:, var_idx]
        y = self.y
        
        for i in range(self.n):
            lhs = x <= x[i]
            rhs = x > x[i]
            
            if rhs.sum() <= self.min_leaf or lhs.sum() <= self.min_leaf:
                continue
            
            lhs_std = y[lhs].std()
            rhs_std = y[rhs].std()
            curr_score = lhs_std * lhs.sum() + rhs_std * rhs.sum()
      
            if curr_score < self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[i]

    def predict(self, x):
        return np.array([self.predict_row(i) for i in x])

    def predict_row(self, x):
        if self.is_leaf:
            return self.val
        
        t = self.lhs if x[self.var_idx] <= self.split else self.rhs
        return t.predict_row(x)
    
    @property
    def split_name(self):
        return self.x.columns[self.var_idx]
    
    @property
    def split_column(self):
        return self.x.values[:, self.var_idx]
    
    @property
    def is_leaf(self):
        return self.score == float('inf')
    
    def __repr__(self):
        ops = f"number of rows: {self.n}, val: {self.val}; "
        if not self.is_leaf:
            ops += f"score: {self.score}, split name: {self.split_name}\n"
        return ops