import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '/Users/manoranjansahoo/Documents/GitHub/DataScienceProjects/HouseValuePrediction/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data
#def rfr_model(X, y):
gsc = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid={
        'max_depth': range(3,7),
        'n_estimators': (10, 50, 100, 1000),
    },
    cv=5, 
    scoring='neg_mean_squared_error', 
    verbose=0,
    n_jobs=-1)

grid_result = gsc.fit(X, y)
best_params = grid_result.best_params_
print(best_params)
rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],                               
                            random_state=False, verbose=False)
    #scores = cross_val_score(rfr, X, y, cv=10, scoring='neg_mean_absolute_error')

    # fit rf_model_on_full_data on all data from the training data
rfr.fit(X,y)

# path to file you will use for predictions
test_data_path = '/Users/manoranjansahoo/Documents/GitHub/DataScienceProjects/HouseValuePrediction/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
#test_data.insert(80,'SalePrice',0)
# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit. 
test_preds = rfr.predict(test_X)
#test_preds = cross_val_predict(rfr,test_X,test_y,cv=10)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
print(output.shape)