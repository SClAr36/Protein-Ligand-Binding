import lightgbm as lgb
import numpy as np
X = np.random.rand(1000, 36)
y = np.random.rand(1000)
dtrain = lgb.Dataset(X, label=y)
lgb.train({"device_type":"cpu"}, dtrain, num_boost_round=10)
print("CPU LightGBM OK")
