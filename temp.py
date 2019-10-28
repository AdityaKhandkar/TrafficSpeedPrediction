# X =  list(range(10))
# y = [x*x for x in X]
#
# from sklearn.model_selection import train_test_split as tts
#
# X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=5)
#
# print("X_train: ",  X_train)
# print("X_test: ", X_test)
# print("y_train: ", y_train)
# print("y_test: ", y_test)

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

speed_data_aug = "speeddata_Aug.csv"
df1 = pd.read_csv(speed_data_aug)

## Maybe add another column week_of_month or day_of_month
df = pd.DataFrame(columns=['target_speed', 'speed_10_mins_ago', 'speed_20_mins_ago',
                            'day_of_week', 'time_of_day', 'route_id', 'avg_speed'])

df.route_id = df1.road_id
df.target_speed = df1.speed
df.speed_10_mins_ago = df1.speed.shift(1)
df.speed_20_mins_ago = df1.speed.shift(2)
df.day_of_week = [(i % 7) + 1 for i in range(len(df))]
df.time_of_day = [("0" if ((i % 144) // 6) < 10 else "") + str((i % 144) // 6) + ":" +
                  ("00" if i % 6 == 0 else str(((i % 144) % 6)*10)) for i in range(len(df))]
df.route_id = df1.road_id

print("before starting")

#
# for times in df['time_of_day'].unique():
#     for ids in df['route_id'].unique():
#         grouped = df.groupby(['time_of_day']).get_group(times).groupby(['route_id']).get_group(ids)
#         n = len(grouped['time_of_day'])
#         mean = grouped['target_speed'].mean()
#         for idx in grouped.index.values:
#             avg_speed = (mean*n - df.loc[idx, 'target_speed'])/(n-1)
#             print("%d, %d" % (idx, avg_speed))
#             temp.loc[idx] = avg_speed

# df['avg_speed'] = temp

# df['avg_speed'] = [(len(df.groupby(['time_of_day']).get_group(times).groupby(['route_id']).get_group(ids)['time_of_day']) *
#                 (df.groupby(['time_of_day']).get_group(times).groupby(['route_id']).get_group(ids)['target_speed'].mean())
#                 - df['target_speed']) /
#                 (len(df.groupby(['time_of_day']).get_group(times).groupby(['route_id']).get_group(ids)['time_of_day']) - 1)
#                 for times in df['time_of_day'].unique() for ids in df['route_id'].unique()]

# print(df.groupby('route_id').get_group(1))

# temp = pd.DataFrame(columns=['avg_speed'])
# temp = pd.read_csv('temp.csv')

for time in df['time_of_day'].unique():
    for id in df['route_id'].unique():
    grouped = df.groupby(['time_of_day']).get_group(time).groupby(['route_id']).get_group(id)
    # print(grouped)
    n = len(grouped['time_of_day'])
    mean = grouped['target_speed'].mean()
    for idx in grouped.index.values:
        avg_speed = (mean*n - df.loc[idx, 'target_speed'])/(n-1)
        # print("%d, %d" % (idx, avg_speed))
        grouped.loc[idx, 'avg_speed'] = avg_speed

# print(grouped)

df.to_csv('customdata_Aug.csv')

# print(df.head(10))
