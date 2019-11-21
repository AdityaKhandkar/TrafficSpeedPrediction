import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn import tree

speed_data_aug = "speeddata_Aug.csv"
custom_data_aug = "customdata_Aug.csv"

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

df0 = pd.read_csv(speed_data_aug)
df = pd.read_csv(custom_data_aug)
df = df.drop(['Unnamed: 0'], axis=1)
df['time_of_day'] = df0['time_id']

df = df.dropna()

X = df.iloc[0:5000, 3:6]
y = df.iloc[0:5000, 0:1]

RANDOM_STATE = 101
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=RANDOM_STATE)
# X_train, X_val, y_train, y_val = tts(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)

regr = RandomForestRegressor(n_estimators=100, max_depth=1000)
regr.fit(X_train, y_train.values.ravel())

# print(regr.score(X_val, y_val.values.ravel()))
# print(clf.feature_importances_)

# features_ = [t.tree_.feature for t in regr.estimators_]
# feature_ = regr.estimators_[0].tree_.feature
# print("feature_ length: ", len(feature_))
# print("feature_: ", feature_)
# print("features_ length: ", len(features_))
# print("features_[0] length:", len(features_[0]))
# print("features_[0]:", features_[0])


def get_attribute_min_depth(tree, target):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature_list = tree.tree_.feature

    def walk(node_id, target_, level):
        # print("node id: %d, feature_[node_id]: %d" % (node_id, feature_list[node_id]))
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


# avg_min_height_0 = math.ceil(average([get_attribute_min_depth(t, 0) for t in regr.estimators_]))
# avg_min_height_1 = math.ceil(average([get_attribute_min_depth(t, 1) for t in regr.estimators_]))
# avg_min_height_2 = math.ceil(average([get_attribute_min_depth(t, 2) for t in regr.estimators_]))

min_height_avgs = [math.ceil(average([get_attribute_min_depth(t, i) for t in regr.estimators_])) for i in range(len(X_train.columns))]

# print(len(X_train.columns)
print(min_height_avgs)

# n_nodes_ = [t.tree_.node_count for t in regr.estimators_]
# children_left_ = [t.tree_.children_left for t in regr.estimators_]
# children_right_ = [t.tree_.children_right for t in regr.estimators_]
# feature_ = [t.tree_.feature for t in regr.estimators_]
# threshold_ = [t.tree_.threshold for t in regr.estimators_]
#
# def explore_tree(estimator, n_nodes, children_left,children_right, feature, threshold,
#                 suffix='', print_tree= False, sample_id=0, feature_names=None):
#
#     if not feature_names:
#         feature_names = feature
#
#
#     assert len(feature_names) == X.shape[1], "The feature names do not match the number of features."
#     # The tree structure can be traversed to compute various properties such
#     # as the depth of each node and whether or not it is a leaf.
#     node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
#     is_leaves = np.zeros(shape=n_nodes, dtype=bool)
#
#     stack = [(0, -1)]  # seed is the root node id and its parent depth
#     while len(stack) > 0:
#         node_id, parent_depth = stack.pop()
#         node_depth[node_id] = parent_depth + 1
#
#         # If we have a test node
#         if (children_left[node_id] != children_right[node_id]):
#             stack.append((children_left[node_id], parent_depth + 1))
#             stack.append((children_right[node_id], parent_depth + 1))
#         else:
#             is_leaves[node_id] = True
#
#     print("The binary tree structure has %s nodes"
#           % n_nodes)
#     if print_tree:
#         print("Tree structure: \n")
#         for i in range(n_nodes):
#             if is_leaves[i]:
#                 print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
#             else:
#                 print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
#                       "node %s."
#                       % (node_depth[i] * "\t",
#                          i,
#                          children_left[i],
#                          feature[i],
#                          threshold[i],
#                          children_right[i],
#                          ))
#             print("\n")
#         print()
#
#     # First let's retrieve the decision path of each sample. The decision_path
#     # method allows to retrieve the node indicator functions. A non zero element of
#     # indicator matrix at the position (i, j) indicates that the sample i goes
#     # through the node j.
#
#     node_indicator = estimator.decision_path(X_test)
#
#     # Similarly, we can also have the leaves ids reached by each sample.
#
#     leave_id = estimator.apply(X_test)
#
#     # Now, it's possible to get the tests that were used to predict a sample or
#     # a group of samples. First, let's make it for the sample.
#
#     #sample_id = 0
#     node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
#                                         node_indicator.indptr[sample_id + 1]]
#
#     print(X_test.iloc[sample_id, :])
#
#     print('Rules used to predict sample %s: ' % sample_id)
#     for node_id in node_index:
#         # tabulation = " "*node_depth[node_id] #-> makes tabulation of each level of the tree
#         tabulation = ""
#         if leave_id[sample_id] == node_id:
#             print("%s==> Predicted leaf index \n"%(tabulation))
#             #continue
#
#         if (X_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]):
#             threshold_sign = "<="
#         else:
#             threshold_sign = ">"
#
#         print("%sdecision id node %s : (X_test[%s, '%s'] (= %s) %s %s)"
#               % (tabulation,
#                  node_id,
#                  sample_id,
#                  feature_names[feature[node_id]],
#                  X_test.iloc[sample_id, feature[node_id]],
#                  threshold_sign,
#                  threshold[node_id]))
#     print("%sPrediction for sample %d: %s"%(tabulation,
#                                           sample_id,
#                                           estimator.predict(X_test)[sample_id]))
#
#     # For a group of samples, we have the following common node.
#     sample_ids = [sample_id, 1]
#     common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
#                     len(sample_ids))
#
#     common_node_id = np.arange(n_nodes)[common_nodes]
#
#     print("\nThe following samples %s share the node %s in the tree"
#           % (sample_ids, common_node_id))
#     print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))
#
#     for sample_id_ in sample_ids:
#         print("Prediction for sample %d: %s"%(sample_id_,
#                                           estimator.predict(X_test)[sample_id_]))
# feature_names = ['day_of_week', 'time_of_day', 'route_id']
# for i,e in enumerate(regr.estimators_):
#
#     print("Tree %d\n"%i)
#     explore_tree(regr.estimators_[i], n_nodes_[i], children_left_[i],
#                  children_right_[i], feature_[i], threshold_[i],
#                  suffix=i, sample_id=1, feature_names=none)
#     print('\n'*2)
