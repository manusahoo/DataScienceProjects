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
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]
#print(X.head)
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_X_valid = pd.DataFrame(my_imputer.transform(val_X))

#print(imputed_X_train.head)

# To improve accuracy, create a new Random Forest model which you will train on all training data

rfr = RandomForestRegressor(n_estimators=10,random_state=0)
rfr.fit(imputed_X_train,train_y)
rfr_train_predictions = rfr.predict(imputed_X_valid)
rfr_val_mae = mean_absolute_error(val_y, rfr_train_predictions)
print(rfr_val_mae)
# path to file you will use for predictions
test_data_path = '/Users/manoranjansahoo/Documents/GitHub/DataScienceProjects/HouseValuePrediction/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]


imputed_X = pd.DataFrame(my_imputer.transform(test_X))

# make predictions which we will submit. 
test_preds = rfr.predict(imputed_X)
#test_preds = cross_val_predict(rfr,test_X,test_y,cv=10)

output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})
print(output.head)