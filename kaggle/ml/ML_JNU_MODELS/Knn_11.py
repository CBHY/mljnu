# import
import numpy as np
import pandas as pd
import sklearn

# load dataset (Planets Dataset)
train = pd.read_csv('D:\CodingData\mljnu\kaggle\ml\datasets\\train_kaggle.csv')
test = pd.read_csv('D:\CodingData\mljnu\kaggle\ml\datasets\\test_kaggle.csv')

# 연관성 없는 단순 숫자 나열 Drop
train = train.drop("index", axis=1).copy()
train = train.drop("number", axis=1).copy()
train = train.drop("year", axis=1).copy()
test = test.drop('index', axis=1).copy()
test = test.drop('number', axis=1).copy()
test = test.drop('year', axis=1).copy()


# "method" 속성을 수치형 데이터로 바꿈
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(train['method'])
train['method'] = encoder.transform(train['method'])

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
test_array = imputer.fit_transform(test)

# np.ndarray to transform pd.DataFrame
RTrain_X = pd.DataFrame(RTrain_X_array, index=RTrain_X.index, columns=RTrain_X.columns).copy()
reg_nan = pd.DataFrame(reg_nan_array, index=reg_nan.index, columns=reg_nan.columns).copy()
test = pd.DataFrame(test_array, index=test.index, columns=test.columns).copy()



# model Selection
from sklearn.neighbors import KNeighborsRegressor

# Training
Knn_Reg = KNeighborsRegressor(n_neighbors=3, weights='distance')
Knn_Reg.fit(RTrain_X, RTrain_y)

# predict and Trans np.ndarray to pd.Series
predict_mass = Knn_Reg.predict(reg_nan)
predict_mass = pd.Series((predict_mass), name='mass').copy()


#IndexRange setting

reg_nan = reg_nan.set_index(pd.Index(range(0, 375))).copy()
predict_mass = pd.Series(predict_mass, index=range(0, 375)).copy()

RTrain_X = RTrain_X.set_index(pd.Index(range(375, 751))).copy()
RTrain_y.index = range(375, 751)

# concat
RTrain_X_y = pd.concat([RTrain_X ,RTrain_y], axis=1).copy()
Preprocessed_NaN = pd.concat([reg_nan ,predict_mass], axis=1).copy()
train = pd.concat([Preprocessed_NaN, RTrain_X_y], axis=0).copy()


# Type Cast
train = train.astype({'method' : 'int64'})
train = train.astype({'orbital_period' : 'float64'})
train = train.astype({'mass' : 'float64'})

test = test.astype({'orbital_period' : 'float64'})
test = test.astype({'mass' : 'float64'})

# Column Settinng
train = train[['method', 'orbital_period', 'mass', 'distance']]
# ________________________________________________________PreProcessing Complete_____________________________________________________


# X, y split
train_X = train.drop('method', axis=1).copy()
train_y = train['method'].copy()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# model selection
ppl = Pipeline([
    ("std", StandardScaler()),
    ("Knn", KNeighborsClassifier(weights='distance', n_neighbors=6))
])


ppl.fit(train_X, train_y)

pred_method = ppl.predict(test)

#pred_method = encoder.inverse_transform(pred_method)

pred_method_df = pd.DataFrame({'method': pred_method})


pred_method_df.to_csv("r3.csv", index=True)


print("Complete")







