import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv("california_housing.csv")

## Without Geographical Info
#X=df.drop(columns = ["MedHouseVal","Latitude","Longitude"])

## With Geographical Info
X=df.drop(columns = ["MedHouseVal"])
y=df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

ridge_model = make_pipeline(
    StandardScaler(),
    Ridge()
)

parameter_grid = {
    "ridge__alpha":[0.01,0.1,1,10,100]
}

ridge_cv = GridSearchCV(
    ridge_model,
    parameter_grid,
    cv = 5,
    scoring = "neg_mean_squared_error"
)

ridge_cv.fit(X_train,y_train)

print("Best alpha:", ridge_cv.best_params_)
print("Best CV MSE:", -ridge_cv.best_score_)

y_pred = ridge_cv.predict(X_test)

print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R2:", r2_score(y_test, y_pred))

