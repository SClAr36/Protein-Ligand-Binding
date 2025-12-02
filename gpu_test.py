import xgboost as xgb
import numpy as np

print("XGBoost version:", xgb.__version__)

X = np.random.rand(20000, 50)
y = np.random.rand(20000)

dtrain = xgb.DMatrix(X, label=y)

params = {
    "tree_method": "hist",   # GPU 3.x 用法
    "device": "cuda",
    "max_depth": 8,
    "eta": 0.1,
    "objective": "reg:squarederror",
}

print("Training...")
bst = xgb.train(params, dtrain, num_boost_round=200)
print("Done.")
