import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import IPython
import graphviz
import re

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute._base import SimpleImputer as Imputer
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from sklearn.ensemble import forest
from sklearn.tree import export_graphviz
from sklearn.metrics import r2_score
from tree_interpretation import *
import numpy as np
import pandas as pd