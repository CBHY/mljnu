# import
import numpy as np
import pandas as pd
import sklearn

# load dataset (Planets Dataset)
train = pd.read_csv('D:\CodingData\mljnu\kaggle\ml\datasets\\train_kaggle.csv')
test = pd.read_csv('D:\CodingData\mljnu\kaggle\ml\datasets\\test_kaggle.csv')

# 연관성 없는 단순 숫자 나열 Drop
train = train.drop("index", axis=1).copy()
test = test.drop('index', axis=1).copy()


# "method" 속성을 수치형 데이터로 바꿈
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(train['method'])
train['method'] = encoder.transform(train['method'])

#--------------------------------------------------------------------------------
# 결측치 처리
reg_train = train[train['mass'].isnull() == False].copy()
reg_nan = train[train['mass'].isnull() == True].copy()
RTrain_X = reg_train.drop("mass", axis=1).copy() # Train X
RTrain_y = reg_train['mass'].copy() # Train y
reg_nan = reg_nan.drop('mass', axis=1).copy()


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
RTrain_X_array = imputer.fit_transform(RTrain_X)
reg_nan_array = imputer.fit_transform(reg_nan)




# np.ndarray to transform pd.DataFrame
RTrain_X = pd.DataFrame(RTrain_X_array, index=RTrain_X.index, columns=RTrain_X.columns).copy()
reg_nan = pd.DataFrame(reg_nan_array, index=reg_nan.index, columns=reg_nan.columns).copy()


RTrain_X = RTrain_X.drop('method', axis=1).copy()
RTrain_X = RTrain_X.drop('year', axis=1).copy()
reg_nan = reg_nan.drop('method', axis = 1).copy()
reg_nan = reg_nan.drop('year', axis = 1).copy()

# Train with Regression model
# model Selection
from sklearn.neighbors import KNeighborsRegressor
Knn_Reg1 = KNeighborsRegressor(n_neighbors=3, weights='distance')
Knn_Reg1.fit(RTrain_X, RTrain_y)

# predict and Trans ndarray to Series
predict_mass = Knn_Reg1.predict(reg_nan)
predict_mass = pd.Series((predict_mass), name='mass').copy()
RTrain_y.index = RTrain_X.index
predict_mass.index = reg_nan.index
train_mass = pd.concat([predict_mass, RTrain_y], axis = 0).copy()
train_mass = train_mass.sort_index().copy()



reg_train2 = train[train['distance'].isnull() == False].copy()
reg_nan2 = train[train['distance'].isnull() == True].copy()
RTrain_X2 = reg_train2.drop("distance", axis=1).copy() # Train X
RTrain_y2 = reg_train2['distance'].copy() # Train y
reg_nan2 = reg_nan2.drop('distance', axis=1).copy()

RTrain_X2_array = imputer.fit_transform(RTrain_X2)
reg_nan2_array = imputer.fit_transform(reg_nan2)
RTrain_X2 = pd.DataFrame(RTrain_X2_array, index=RTrain_X2.index, columns=RTrain_X2.columns).copy()
reg_nan2 = pd.DataFrame(reg_nan2_array, index=reg_nan2.index, columns=reg_nan2.columns).copy()

RTrain_X2 = RTrain_X2.drop('number', axis=1).copy()
RTrain_X2 = RTrain_X2.drop('orbital_period', axis=1).copy()
reg_nan2 = reg_nan2.drop('number', axis = 1).copy()

reg_nan2 = reg_nan2.drop('orbital_period', axis = 1).copy()

Knn_Reg2 = KNeighborsRegressor(n_neighbors=3, weights='distance')
Knn_Reg2.fit(RTrain_X2, RTrain_y2)

# predict and Trans ndarray to Series
predict_distance = Knn_Reg2.predict(reg_nan2)
predict_distance = pd.Series((predict_distance), name='distance').copy()
RTrain_y2.index = RTrain_X2.index
predict_distance.index = reg_nan2.index
train_distance = pd.concat([predict_distance, RTrain_y2], axis = 0).copy()
train_distance = train_distance.sort_index().copy()




train = train.drop('distance', axis = 1).copy()
train = train.drop('mass', axis = 1).copy()



# concat mass
train = pd.concat([train , train_distance], axis=1).copy()
train = pd.concat([train, train_mass], axis=1).copy()


# orbital_period NaN Processing
train_array = imputer.fit_transform(train)
train = pd.DataFrame(train_array, index=train.index, columns=train.columns).copy()


# 결측치 처리

reg_test = test[test['mass'].isnull() == False].copy()
reg_nant = test[test['mass'].isnull() == True].copy()
Rtest_X = reg_test.drop("mass", axis=1).copy() # test X
Rtest_y = reg_test['mass'].copy() # test y
reg_nant = reg_nant.drop('mass', axis=1).copy()

reg_nant = reg_nant.drop("year", axis=1).copy()


reg_nant_array = imputer.fit_transform(reg_nant)
reg_nant = pd.DataFrame(reg_nant_array, index=reg_nant.index, columns=reg_nant.columns).copy()


# predict and Trans ndarray to Series
predict_mass_test = Knn_Reg1.predict(reg_nant)
predict_mass_test = pd.Series((predict_mass_test), name='mass').copy()
predict_mass_test.index = reg_nant.index
test_mass = pd.concat([predict_mass_test, Rtest_y], axis = 0).copy()
test_mass = test_mass.sort_index().copy()



# predict and Trans ndarray to Series
test = test.drop('mass', axis = 1).copy()


# concat mass
test = pd.concat([test, test_mass], axis=1).copy()


# orbital_period NaN Processing
test_array = imputer.fit_transform(test)
test = pd.DataFrame(test_array, index=test.index, columns=test.columns).copy()


# Type Cast
train = train.astype({'number' : 'int64'})
train = train.astype({'method' : 'int64'})
train = train.astype({'orbital_period' : 'float64'})
train = train.astype({'mass' : 'float64'})
train = train.astype( {'year' : 'int64'})

test = test.astype({'number' : 'int64'})
test = test.astype({'orbital_period' : 'float64'})
test = test.astype({'mass' : 'float64'})
test = test.astype({'year' : 'int64'})

# Column Settinng
train = train[['method' ,'number', 'orbital_period', 'mass', 'distance', 'year']]
# ________________________________________________________PreProcessing Complete_____________________________________________________
# ________________________________________________________PreProcessing Complete_____________________________________________________
print(train.info())
print(test.info())


# X, y split
train_X = train.drop('method', axis=1).copy()
train_y = train['method'].copy()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

train_X = train_X.drop('number', axis=1).copy()
train_X = train_X.drop('mass', axis=1).copy()
test = test.drop('mass', axis=1).copy()
test = test.drop('number', axis=1).copy()



# model selection
ppl = Pipeline([
    ("std", StandardScaler()),
    ("Knn", KNeighborsClassifier(weights='distance', n_neighbors=6))
])


ppl.fit(train_X, train_y)

pred_method = ppl.predict(test)

pred_method = encoder.inverse_transform(pred_method)

pred_method_df = pd.DataFrame({'method': pred_method})


pred_method_df.to_csv("Knn_12.csv", index=True)


print("Complete")







