import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_attribute_min_depth(tree, target):
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature_list = tree.tree_.feature

    def walk(node_id, target_, level):
        if feature_list[node_id] == target_:
            return level
        if children_left[node_id] != children_right[node_id]:
            left_min = walk(children_left[node_id], target_, level + 1)
            right_min = walk(children_right[node_id], target_, level + 1)

            if left_min > -1 and right_min > -1:
                return min(left_min, right_min)

            return max(left_min, right_min)
        else: # leaf
            return -1

    root_node_id = 0
    return walk(root_node_id, target, 0)


def average(lst):
    return sum(lst) / len(lst)


pd.set_option('max_colwidth', -1)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

speed_data_aug = "speeddata_Aug.csv"
custom_data_aug = "customdata_Aug.csv"

df0 = pd.read_csv(speed_data_aug)
df = pd.read_csv(custom_data_aug)

df = df.drop(['Unnamed: 0'], axis=1)
df['time_of_day'] = df0['time_id']

df = df.dropna()

X = df.iloc[0:10000, 1:]
y = df.iloc[0:10000, 0:1]

RANDOM_STATE = 101
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = tts(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)

# X_train_norm = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
# X_val_norm = pd.DataFrame(scaler.fit_transform(X_val.values), columns=X_val.columns, index=X_val.index)

scalerX = StandardScaler().fit(X_train)
X_train_norm = scalerX.transform(X_train)
X_val_norm = scalerX.transform(X_val)
X_test_norm = scalerX.transform(X_test)

# scalerY = StandardScaler().fit(y_train)
# y_train_norm = scalerY.transform(y_train)
# y_val_norm = scalerY.transform(y_val)
# y_test_norm = scalerY.transform(y_test)

# X_train_norm[:, 0] = X_train_norm[:, 0]*2
# print("X_train_norm:", X_train_norm[:, 0])
# print("X_train_norm:", X_train_norm)
# print("X_val_norm:", X_val_norm)
# print("y_train_norm:", y_train_norm)
# print("y_val_norm:", y_val_norm)

# svr_before = svm.SVR()
# svr_before.fit(X_train_norm, y_train.values.ravel())
# accuracy = svr_before.score(X_val_norm, y_val.values.ravel())
# print('accuracy1 = ', accuracy)

max_depth = 50
regr = RandomForestRegressor(n_estimators=100, max_depth=max_depth)
regr.fit(X_train_norm, y_train.values.ravel())

min_height_avgs = [math.ceil(average([get_attribute_min_depth(t, i) for t in regr.estimators_]))
                   for i in range(len(X_train.columns))]
print(min_height_avgs)

# for i in range(len(X_train.columns)):
#     scaler = MinMaxScaler(feature_range=(0, 2*(max_depth - min_height_avgs[i])))
#     X_scaled = X_train_norm[:, i].reshape(-1, 1)
#     X_scaled = scaler.fit_transform(X_scaled)
#     X_train_norm[:, i] = X_scaled.reshape(X_train_norm[:, i].shape)
#
#     X_scaled = X_val_norm[:, i].reshape(-1, 1)
#     scaler.fit(X_scaled)
#     X_val_norm[:, i] = X_scaled.reshape(X_val_norm[:, i].shape)
#     # X_test_norm[:, i] = scaler.fit(X_test_norm[:, i])
#
# scaler = MinMaxScaler(feature_range=(0, 2*(max_depth - math.ceil(average(min_height_avgs)))))
# y_train_norm = scaler.fit_transform(y_train_norm)
# scaler.fit(y_val_norm)

X_train_norm1 = X_train_norm
X_val_norm1 = X_val_norm

for i in range(len(X_train.columns)):
    X_train_norm1[:, i] = X_train_norm1[:, i] * (min_height_avgs[i])
    X_val_norm1[:, i] = X_val_norm1[:, i] * (min_height_avgs[i])
    # X_test_norm[:, i] = X_test_norm[:, i] * min_height_avgs[i] / 2

svr_after = svm.SVR()
svr_after.fit(X_train_norm1, y_train.values.ravel())
accuracy = svr_after.score(X_val_norm1, y_val.values.ravel())
print('accuracy1 = ', accuracy)


for i in range(len(X_train.columns)):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = X_train_norm[:, i].reshape(-1, 1)
    X_scaled = scaler.fit_transform(X_scaled)
    X_train_norm[:, i] = X_scaled.reshape(X_train_norm[:, i].shape)

    X_scaled = X_val_norm[:, i].reshape(-1, 1)
    scaler.fit(X_scaled)
    X_val_norm[:, i] = X_scaled.reshape(X_val_norm[:, i].shape)
    # X_test_norm[:, i] = scaler.fit(X_test_norm[:, i])

scaler = MinMaxScaler(feature_range=(0, 1))
y_train_norm = scaler.fit_transform(y_train)
scaler.fit(y_val)

svr_after = svm.SVR()
svr_after.fit(X_train_norm, y_train.values.ravel())
accuracy = svr_after.score(X_val_norm, y_val.values.ravel())
print('accuracy2 = ', accuracy)

# for i in range(len(X_train.columns)):
#     scaler = MinMaxScaler(feature_range=(0, 2**(max_depth - min_height_avgs[i])))
#     X_scaled = scaler.fit_transform(np.array(X_train_norm[:, i]).reshape(-1, 1))
#     X_train_norm[:, i] = X_scaled.reshape(X_train_norm[:, i].shape)
#
#     X_scaled = np.array(X_val_norm[:, i]).reshape(-1, 1)
#     scaler.fit(X_scaled)
#     X_val_norm[:, i] = X_scaled.reshape(X_val_norm[:, i].shape)
#     # scaler.fit(X_test.iloc[:, i])
#
# scaler = MinMaxScaler(feature_range=(0, 1))
# y_train = scaler.fit_transform(y_train)
# scaler.fit(y_val)

# def normalize(DF):
#     result = DF.copy()
#     for feature_name in DF.columns:
#         max_value = 2**(max_depth - min_height_avgs[DF.columns.get_loc(feature_name)])
#         min_value = 0
#         result[feature_name] = (DF[feature_name] - min_value) / (max_value - min_value)
#     return result
#
#
# X_train_norm = normalize(X_train_norm)
#
# X_val_norm = normalize(X_val_norm)





## Accuracy by using everything except avg_speed as features is .5802148

# plt.scatter(X_val, y_val, color = 'magenta')
# fig, ax = plt.subplots()
# # plt.xlim(0, 5000)
# # plt.ylim(0, 1000)
# plt.scatter(df['target_speed'][0:5000], df['time_of_day'][0:5000])
# plt.title('Speed vs time (Support Vector Regression Model)')
# plt.xlabel('speed')
# plt.ylabel('time')
# every_nth = 4
# for n, label in enumerate(ax.xaxis.get_ticklabels()):
#     if n % every_nth != 0:
#         label.set_visible(False)
# plt.show()
