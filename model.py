import pandas as pd
import numpy as np
from sklearn import linear_model
# from sklearn import ensemble
# from sklearn import neural_network
# from sklearn import svm
from sklearn import tree
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("DBS_SingDollar.csv", encoding='utf-8')
print(df)

X = df['SGD'].values.reshape(-1,1)
y = df['DBS'].values.reshape(-1,1)
model = linear_model.LinearRegression()
model.fit(X, y)
pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, pred))
print(rmse)
joblib.dump(model, "regression")

model = tree.DecisionTreeRegressor()
model.fit(X, y)
pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, pred))
print(rmse)
joblib.dump(model, "tree")

# model = ensemble.RandomForestRegressor()
# model.fit(X, y)
# pred = model.predict(X)
# rmse = np.sqrt(mean_squared_error(y, pred))
# print(rmse)

# model = ensemble.GradientBoostingRegressor()
# model.fit(X, y)
# pred = model.predict(X)
# rmse = np.sqrt(mean_squared_error(y, pred))
# print(rmse)

# model = neural_network.MLPRegressor(solver='lbfgs', hidden_layer_sizes=(50,50), max_iter=500)
# model.fit(X, y)
# pred = model.predict(X)
# rmse = np.sqrt(mean_squared_error(y, pred))
# print(rmse)

# model = svm.SVR()
# model.fit(X, y)
# pred = model.predict(X)
# rmse = np.sqrt(mean_squared_error(y, pred))
# print(rmse)