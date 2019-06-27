import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '/Users/manoranjansahoo/Documents/GitHub/DataScienceProjects/HouseValuePrediction/train.csv'
home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y_train = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X_train = home_data[features]

# path to file you will use for predictions
test_data_path = '/Users/manoranjansahoo/Documents/GitHub/DataScienceProjects/HouseValuePrediction/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
X_valid = test_data[features]

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# To improve accuracy, create a new Random Forest model which you will train on all training data
rfr = RandomForestRegressor(max_depth=10, n_estimators=100,random_state=0)

scores = -1 * cross_val_score(rfr, imputed_X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

rfr.fit(imputed_X_train,y_train)
rfr_train_predictions = rfr.predict(imputed_X_valid)




#rfr_val_mae = mean_absolute_error(val_y, rfr_train_predictions)
#print(rfr_val_mae)

# make predictions which we will submit. 

output = pd.DataFrame({'Id': test_data.Id,'SalePrice': rfr_train_predictions})
print(output.head)
