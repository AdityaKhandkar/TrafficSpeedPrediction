import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

speed_data_aug = "C:/Aditya/College/Penn State/Semesters/5) Fall 2019/Master's Paper/Data sets/DataSet/speeddata_Aug.csv"

RANDOM_STATE = 101

df1 = pd.read_csv(speed_data_aug)
df2 = pd.read_csv(speed_data_aug)

df1 = df1[df1.road_id == 1]
# df1 = df1[df1.day_id == 1]
df1 = df1.drop('road_id', axis=1)

X = df1.iloc[:, 2:3].values.astype(float)
y = df1.iloc[:, 1:2].values.astype(float)

print(len(X), len(y))

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = tts(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)

regressor = svm.SVR(kernel='poly', degree=2, C=50, epsilon=0.02)
regressor.fit(X_train, y_train)
accuracy = regressor.score(X_val, y_val)
print('accuracy = ', accuracy)

plt.scatter(X_val, y_val, color = 'magenta')
plt.plot(X_val, regressor.predict(X_val), color ='green')
plt.title('Time vs speed (Support Vector Regression Model)')
plt.xlabel('speed')
plt.ylabel('time')
plt.show()
