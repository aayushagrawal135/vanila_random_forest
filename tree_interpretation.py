from imports import *

class PredictionConfidence:
    def __init__(self, forest_model, to_be_predicted_rows, predictions=None):
        self.forest_model = forest_model
        self.to_be_predicted_rows = to_be_predicted_rows
        if predictions is None:
            self.predictions = np.stack([t.predict(to_be_predicted_rows.values) for t in forest_model.trees])
        else:
            self.predictions = predictions

    def cumulative_confidence(self):
        return self.std_vals / self.mean_vals

    def feature_confidence(self, cols):
        conf = self.cumulative_confidence()
        x = self.to_be_predicted_rows.copy()
        x["lack of confidence"] = list(conf)
        return x.groupby(cols).agg({"lack of confidence": "mean"}).reset_index()

    @property
    def mean_vals(self):
        return np.mean(self.predictions, axis=0)

    @property
    def std_vals(self):
        return np.std(self.predictions, axis=0)


class FeatureImportance:
    def __init__(self, model, x, y):
        self.x = x
        self.y = y
        self.feat_imp = dict()
        self.model = model
        self.pre_shuffle_score = r2_score(y, model.predict(x.values))

    @property
    def get_feature_importance(self):
        ks = list(self.feat_imp.keys())
        vals = list(self.feat_imp.values())
        df = pd.DataFrame(vals, index=ks, columns=["feature importance"])
        return df.sort_values("feature importance", ascending=False)

    def find_importance_for_all(self):
        for col in self.x.columns:
            x = self.shuffle(col)
            self.find_importance_for_each(x, col)

    def find_importance_for_each(self, x, col):
        score = r2_score(self.y, self.model.predict(x.values))
        self.feat_imp[col] = self.pre_shuffle_score - score

    def shuffle(self, col):
        x = self.x.copy()
        x_col = x[col]
        x_vals = x_col.reindex(np.random.permutation(x_col.index))
        x[col] = x_vals.values
        return x


class PartialDependence:
    def __init__(self, model, x):
        self.model = model
        self.x = x

    def agg_dependence(self, feature):
        vals = self.unbiased_row_dependence(feature)
        mean_vals = np.mean(vals, axis=1)
        return pd.DataFrame(mean_vals, index=set(self.x[feature]))

    def row_granular_dependence(self, feature):
        vals = self.unbiased_row_dependence(feature)
        return pd.DataFrame(vals, index=set(self.x[feature]))

    def unbiased_row_dependence(self, feature):
        row_grans = list()
        for i in set(self.x[feature].values):
            x = self.x.copy()
            x[feature] = i
            vals = self.model.predict(x.values)
            row_grans.append(vals)
        return row_grans


class RowWaterfall:
    def __init__(self, row, columns, model):
        self.row = row
        self.model = model
        self.waterfall = dict()
        for c in columns:
            self.waterfall[c] = 0

    def total_change_model(self):
        for tree in self.model.trees:
            self.total_change_tree(tree)

        for key in self.waterfall:
            self.waterfall[key] = self.waterfall[key] / self.model.n_trees

        vals = list(self.waterfall.values())
        keys = list(self.waterfall.keys())
        wf = pd.DataFrame(vals, index=keys, columns=["waterfall contribs"])
        return wf.sort_values("waterfall contribs", ascending=False)

    def total_change_tree(self, tree):
        if tree.is_leaf:
            return

        split_name = tree.split_name
        split = tree.split

        sub_tree = tree.lhs if self.row[split_name] <= split else tree.rhs

        gain = sub_tree.val - tree.val
        self.waterfall[split_name] = self.waterfall[split_name] + gain

        return self.total_change_tree(sub_tree)