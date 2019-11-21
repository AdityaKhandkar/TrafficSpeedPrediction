import math
import numpy as np
import pandas as pd
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
X_train, X_val, y_train, y_val = tts(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)

X_train_1, X_train_2, X_train_3 = X_train, X_train, X_train
X_val_1, X_val_2, X_val_3 = X_val, X_val, X_val
X_test_1, X_test_2, X_test_3 = X_test, X_test, X_test
y_train_1, y_train_2, y_train_3 = y_train, y_train, y_train
y_val_1, y_val_2, y_val_3 = y_val, y_val, y_val
y_test_1, y_test_2, y_test_3 = y_test, y_test, y_test

scalerX = StandardScaler().fit(X_train_1)
X_train_1_norm = scalerX.transform(X_train_1)
X_val_1_norm = scalerX.transform(X_val_1)
X_test_1_norm = scalerX.transform(X_test_1)

scalerY = StandardScaler().fit(y_train_1)
y_train_1_norm = scalerY.transform(y_train_1)
y_val_1_norm = scalerY.transform(y_val_1)
y_test_1_norm = scalerY.transform(y_test_1)


svr_1 = svm.SVR()
svr_1.fit(X_train_1_norm, y_train_1.values.ravel())
accuracy = svr_1.score(X_val_1_norm, y_val_1.values.ravel())
print('accuracy1 = ', accuracy)

max_depth = 50
regr = RandomForestRegressor(n_estimators=100, max_depth=max_depth)
regr.fit(X_train_1_norm, y_train_1_norm.ravel())

min_height_avgs = [math.ceil(average([get_attribute_min_depth(t, i) for t in regr.estimators_]))
                   for i in range(len(X_train.columns))]
print(min_height_avgs)

for i in range(len(X_train.columns)):
    X_train_1_norm[:, i] = X_train_1_norm[:, i] * (max_depth - min_height_avgs[i])
    X_val_1_norm[:, i] = X_val_1_norm[:, i] * (max_depth - min_height_avgs[i])
    # X_test_norm[:, i] = X_test_norm[:, i] * min_height_avgs[i] / 2

svr_1 = svm.SVR()
svr_1.fit(X_train_1_norm, y_train_1.values.ravel())
accuracy = svr_1.score(X_val_1_norm, y_val_1.values.ravel())
print('accuracy2 = ', accuracy)


# with X_train_2
scaler = MinMaxScaler(feature_range=(0,1))
X_train_2 = pd.DataFrame(scaler.fit_transform(X_train_2.values), columns=X_train.columns, index=X_train.index)
X_val_2 = pd.DataFrame(scaler.fit_transform(X_val_2.values), columns=X_val.columns, index=X_val.index)

y_train_2 = pd.DataFrame(scaler.fit_transform(y_train_2.values), columns=y_train.columns, index=y_train.index)
y_val_2 = pd.DataFrame(scaler.fit_transform(y_val_2.values), columns=y_val.columns, index=y_val.index)

svr_2 = svm.SVR()
svr_2.fit(X_train_2, y_train_2)
accuracy = svr_2.score(X_val_2, y_val_2)
print('accuracy3 = ', accuracy)

# with X_train_3
# max_depth = 50
# regr = RandomForestRegressor(n_estimators=100, max_depth=max_depth)
# regr.fit(X_train_3, y_train_3.values.ravel())
#
# min_height_avgs = [math.ceil(average([get_attribute_min_depth(t, i) for t in regr.estimators_]))
#                    for i in range(len(X_train.columns))]
# print(min_height_avgs)
#
# for i in range(len(X_train.columns)):
#     X_train_3.iloc[:, i] = X_train_3.iloc[:, i] * (min_height_avgs[i] / 2)
#     X_val_3.iloc[:, i] = X_val_3.iloc[:, i] * (min_height_avgs[i] / 2)
#     # X_test_norm[:, i] = X_test_norm[:, i] * min_height_avgs[i] / 2
#
# svr_3 = svm.SVR()
# svr_3.fit(X_train_3, y_train_3.values.ravel())
# accuracy = svr_1.score(X_val_3, y_val_3.values.ravel())
# print('accuracy3 = ', accuracy)
