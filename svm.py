import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split as tts

speed_data_aug = "speeddata_Aug.csv"
custom_data_aug = "customdata_Aug.csv"

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

df0 = pd.read_csv(speed_data_aug)
df = pd.read_csv(custom_data_aug)
df = df.drop(['Unnamed: 0'], axis=1)
df['time_of_day'] = df0['time_id']

print(df.head(150))

df = df.dropna()

X = df.iloc[:, 1:6] #.values.astype(float)
y = df.iloc[:, 0:1] #.values.astype(float)

# print(X[:10])
# print(y[:10])

RANDOM_STATE = 101
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = tts(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)

svr = svm.SVR()
svr.fit(X_train, y_train.values.ravel())
accuracy = svr.score(X_val, y_val)
print('accuracy = ', accuracy)

# plt.scatter(X_val, y_val, color = 'magenta')
# plt.plot(X_val, svr.predict(X_val), color ='green')
# plt.title('Time vs speed (Support Vector Regression Model)')
# plt.xlabel('speed')
# plt.ylabel('time')
# plt.show()
