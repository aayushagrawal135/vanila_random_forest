from imports import *
from fast import *
from tree_ensemble import *

# Load quick data
RAW_PATH = "../../../../repos/fastai/courses/ml1/tmp/bulldozers-raw"
df_raw = pd.read_feather(RAW_PATH)

# Process the data
df_trn, y_trn, nas = proc_df(df_raw, "SalePrice")

# splitting dataframe
def split_vals(a, n):
    return a[:n], a[n:]

n_valid = 300
n_trn = len(df_raw) - n_valid
x_train, x_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)

cols = ['MachineID', 'YearMade', 'MachineHoursCurrentMeter', 'ProductSize', 'Enclosure',
        'Coupler_System', 'saleYear']

x_sub = x_train[cols]
x_valid_sub = x_valid[cols]
sample_sz = 1000
min_leaf = 5
n_trees = 3

sample_forest = TreeEnsemble(X = x_sub, Y = y_train, n_trees = n_trees, sample_sz = sample_sz, min_leaf = min_leaf)
preds = sample_forest.predict(x_valid_sub.values)
print(np.shape(preds))

"""
pcf = PredictionConfidence(forest_model = sample_forest, to_be_predicted_rows = x_valid_sub)
ym = pcf.feature_confidence(["YearMade"])
ym.sort_values("lack of confidence")
feat_conf = pcf.cumulative_confidence()
np.shape(feat_conf)
"""

"""
f = FeatureImportance(sample_forest, x_valid_sub, y_valid)
f.find_importance_for_all()
print(f.get_feature_importance)
"""

"""
pdp = PartialDependence(sample_forest, x_valid_sub)
df = pdp.agg_dependence("YearMade")
print(df)
"""

row = x_valid_sub.iloc[0, :]
columns = x_valid_sub.columns
rw = RowWaterfall(row, columns, sample_forest)
wf = rw.total_change_model()
print(wf)

"""
print(r2_score(y_valid, preds))

m = RandomForestRegressor(n_estimators=1, min_samples_leaf=5, bootstrap=False)
samp_x = sample_forest.trees[0].x
samp_y = sample_forest.trees[0].y
m.fit(samp_x, samp_y)
m_preds = m.predict(x_valid_sub.values)
print(r2_score(y_valid, m_preds))
"""