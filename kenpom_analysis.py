# imports
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# read xlsx file
filepath = 'SET FILEPATH' # for your device
bball_df = pd.read_excel(filepath + "KenPom.xlsx")

# arg: text, output: list of string and integer
def split_by_integer(text):
    match = re.split(r'(\d+)', text)
    
    if match:
        return match
    return [text, '']
name_df = bball_df['Team']
# correct dataframe without march madness ranking
name_df = name_df.apply(lambda x: pd.Series(split_by_integer(x)[0]))


# drop conference, year, name, multicollinear factors
bball_df.drop(columns = ['Team', 'Conf', 'Year', 'NetRtg', 'ORtg', 'DRtg', 'AdjT', 'Luck', 'Str_NetRtg', 'NCSOS_NetRtg'],  inplace = True)
bball_df.dropna(inplace = True)
# drop other columns

# kNN
# predict ranking
x = bball_df.drop(columns = 'Rk', axis = 1)
y = bball_df['Rk']
z = bball_df['Win Championship']

# scaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

# pull test and train data
# Rank Set
x1_train, x1_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    random_state = 0,
                                                    stratify = y)

# Win Set
x2_train, x2_test, z_train, z_test = train_test_split(x,
                                                     z,
                                                     random_state = 0,
                                                     stratify = z)

# different k parameters
# Plug in to determine optimal accuracy
min_k = 3
max_k = 20

# rank set loop
rank_acc = 0
rank_val = 0
for i in range(min_k, max_k):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(x1_train, y_train)
    # find acc
    accuracy = neigh.score(x1_test, y_test)
    # print(f'Rank accuracy: {accuracy} k: {i}')
    if accuracy > rank_acc:
        rank_acc = accuracy
        rank_val = i

# win loop
win_acc = 0
win_val = 0
for i in range(min_k, max_k):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(x2_train, z_train)
    # find acc
    accuracy = neigh.score(x2_test, z_test)
    # print(f'Champ accuracy: {accuracy} k: {i}')
    if accuracy > win_acc:
        win_acc = accuracy
        win_val = i

print(f'Accuracy for Rank is {rank_acc} with optimal k = {rank_val}')
print(f'Accuracy for Win Championship is {win_acc} with optimal k = {win_val}') # prob only 1 winner in test set, probability issue

# Next: Test set on most recent kenpom data
bball2_df = pd.read_excel(filepath + "KenPom2025.xlsx")
bball2_df.dropna(inplace = True)
teamlist = bball2_df['Team']
bball2_df.drop(columns = ['Team', 'Conf', 'Year', 'NetRtg', 'ORtg', 'DRtg', 'AdjT', 'Luck', 'Str_NetRtg', 'NCSOS_NetRtg', 'Rk'],  inplace = True)

# scale
scaler.fit(bball2_df)
bball2_df = scaler.transform(bball2_df)


# prediction

neigh = KNeighborsClassifier(n_neighbors = rank_val)
neigh.fit(x1_train, y_train)

# determine closest ranking by using test set
q = neigh.predict(bball2_df)


# same with win
neigh = KNeighborsClassifier(n_neighbors = win_val)
neigh.fit(x2_train, z_train)
p = neigh.predict(bball2_df)

pred_df = pd.DataFrame([q, p], columns= teamlist.to_numpy())

pred_df.to_csv('data.csv')






    
    
    




