#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn import tree
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Predictions')
parser.add_argument('-p', '--path_to_train_dataset', default='train.csv', type=str)
parser.add_argument('-t', '--path_to_test_dataset', default='test.csv', type=str)
parser.add_argument('-o', '--out_path_to_csv_file', default='predictions.csv', type=str)
args = parser.parse_args()

X = pd.read_csv(args.path_to_train_dataset, sep=',')
X = X.drop('id', axis=1)
Y = X.copy()

for i, col in enumerate(X.columns):
    Y = Y.drop([X.columns[i]], axis=1)
    if i == 8:
        break
        
# Преобразование значений столбцов
## Дата 

season = [] 
day_month = [] 
months = [] 
year = []
for i, x in enumerate(X['Дата']):
    day = int(x[:2])
    month = int(x[3:5])
    day_month.append(day)
    months.append(month)
    season.append(month // 3 % 4 + 1)
    year.append(int(x[-4:]))


df = pd.DataFrame(season, columns=['season'])
df['day_month'] = day_month
df['months'] = months
df['year'] = year

# # Время
# 
# ### с 4 до 11 часов - 1
# ### с 11 до 17 часов - 2
# ### с 17 до 22 часов - 3
# ### с 22 до 4 часов - 4


times_of_day = []
hours = []
minutes = []
dict_times = {1: 4,         2: 4,         3: 4,         4: 1,
              5: 1,         6: 1,         7: 1,         8: 1,
              9: 1,        10: 1,        11: 2,        12: 2,
             13: 2,        14: 2,        15: 2,        16: 2,
             17: 3,        18: 3,        19: 3,        20: 3,
             21: 3,        22: 4,        23: 4,         0: 4}

for i, x in enumerate(X['Время']):
    times_of_day.append(dict_times[int(x[:2])])
    hours.append(int(x[:2]))
    minutes.append(int(x[3:5]))
    
df['times_of_day'] = times_of_day  
df['hours'] = hours  
df['minutes'] = minutes


# # Зависимость вида ДТП от времени суток
count = [[0 for i in range(24)] for j in range(18)]
# dict_ = {}
# names = pd.unique(X['Вид ДТП'])
# for i, name in enumerate(names):
#     dict_[name] = i

dict_acc = {'Столкновение':18,
         'nan':17,
         'Наезд на препятствие':16,
         'Наезд на пешехода':15,
         'Наезд на стоящее ТС':14,
         'Опрокидывание':13,
         'Съезд с дороги':12,
         'Наезд на велосипедиста':11,
         'Иной вид ДТП':10,
         'Падение пассажира':9,
         'Наезд на животное':8,
         'Отбрасывание предмета (отсоединение колеса)':7,
         'Наезд на внезапно возникшее препятствие':6,
         'Падение груза':5,
         'Наезд на лицо, не являющееся участником дорожного движения (иного участника ДТП), осуществляющее производство работ':4,
         'Наезд на лицо, не являющееся участником дорожного движения (иного участника ДТП), осуществляющее какую-либо другую деятельность':3,
         'Возгорание вследствие технической неисправности движущегося или остановившегося ТС, участвующего в дорожном движении':2,
         'Наезд на лицо, не являющееся участником дорожного движения (иного участника ДТП), осуществляющее несение службы':1
        }
names = list(dict_acc.keys())
accidents = []

for i, [x1, x2] in enumerate(zip(X['Вид ДТП'], X['Время'])):
    n = dict_acc[str(x1)]
    m = int(x2[:2])
    count[len(count) - n][m] += 1
    accidents.append(n)


accident_time = []
for i, arr in enumerate(count):
    x = np.arange(0, 24)
    y = arr
    accident_time.append(np.array(arr))
    

dang = []
for i, (x1, x2) in enumerate(zip(X['Вид ДТП'], X['Время'])):
    hour = int(x2[:2])
    acc_name = names.index(str(x1))
    dang.append(accident_time[acc_name][hour])

df['dang'] = dang
df['accidents'] = accidents


# ## Место
dict_scene = dict(X['Место'].value_counts())
values = list(dict_scene.values())
values = np.array(values) 

for i, name in enumerate(list(dict_scene.keys())):
    dict_scene[name] = values[i]

scene_values = []
for i, name in enumerate(X['Место']):
    scene_values.append(dict_scene[name])
    
df['scene_values'] = scene_values


# ## Улица
dict_street = dict(X['Улица'].value_counts())
values = list(dict_street.values())

for i, name in enumerate(list(dict_street.keys())):
    dict_street[name] = values[i]
    
street_values = []
for i, name in enumerate(X['Улица']):
    if str(name) != 'nan':
        street_values.append(dict_street[name])
    else:
        street_values.append(0)
    
df['street_values'] = street_values

# ## Дом
dict_house = dict(X['Дом'].value_counts())
values = list(dict_house.values())

for i, name in enumerate(list(dict_house.keys())):
    dict_house[name] = values[i]
    
house_values = []
for i, name in enumerate(X['Дом']):
    if str(name) != 'nan':
        house_values.append(dict_house[name])
    else:
        house_values.append(0)
    
df['house_values'] = house_values

# ## Дорога
dict_road = dict(X['Дорога'].value_counts())
values = list(dict_road.values())
values = np.array(values)

for i, name in enumerate(list(dict_road.keys())):
    dict_road[name] = values[i]

road_values = []
for i, name in enumerate(X['Дорога']):
    if str(name) != 'nan':
        road_values.append(dict_road[name])
    else:
        road_values.append(0)
    
df['road_values'] = road_values

# ## Километр
dict_km = dict(X['Километр'].value_counts())
values = list(dict_km.values())

for i, name in enumerate(list(dict_km.keys())):
    dict_km[name] = values[i]
    
km_values = []
for i, name in enumerate(X['Километр']):
    if str(name) != 'nan':
        km_values.append(dict_km[name])
    else:
        km_values.append(0)
    
df['km_values'] = km_values

# ## Метр
dict_meter = dict(X['Метр'].value_counts())
values = list(dict_meter.values())

for i, name in enumerate(list(dict_meter.keys())):
    dict_meter[name] = values[i]
    
meter_values = []
for i, name in enumerate(X['Метр']):
    if str(name) != 'nan':
        meter_values.append(dict_meter[name])
    else:
        meter_values.append(0)
    
df['meter_values'] = meter_values

# Просмотр минимального и максимального значения каждого столбца
# for col in df.columns[:]:
#     col = df[col]
#     print(col.min(), '\t', col.max())



# # Train
df_train = df.copy()
x_train, y_train = df_train, Y
# print(x_train.shape, y_train.shape)

clf = tree.DecisionTreeClassifier(max_depth=24, random_state=81)
clf = clf.fit(x_train, y_train)


# # Подготовка тестового набора
T = pd.read_csv(args.path_to_test_dataset, sep=',')
season = [] 
day_month = [] 
months = [] 
year = []
for i, x in enumerate(T['Дата']):
    day = int(x[:2])
    month = int(x[3:5])
    day_month.append(day)
    months.append(month)
    season.append(month // 3 % 4 + 1)
    year.append(int(x[-4:]))
    
df = pd.DataFrame(season, columns=['season'])
df['day_month'] = day_month
df['months'] = months
df['year'] = year

times_of_day = []
hours = []
minutes = []


for i, x in enumerate(T['Время']):
    times_of_day.append(dict_times[int(x[:2])])
    hours.append(int(x[:2]))
    minutes.append(int(x[3:5]))
    
df['times_of_day'] = times_of_day  
df['hours'] = hours  
df['minutes'] = minutes


accidents = []
dang = []

for i, (x1, x2) in enumerate(zip(T['Вид ДТП'], T['Время'])):
    n = dict_acc[str(x1)]
    accidents.append(n)
    
    hour = int(x2[:2])
    acc_name = names.index(str(x1))
    dang.append(accident_time[acc_name][hour])

df['dang'] = dang
df['accidents'] = accidents


scene_values = []
for i, name in enumerate(T['Место']):
    try:
        if str(name) != 'nan':
            scene_values.append(dict_scene[name])
        else:
            scene_values.append(0)
    except:
        scene_values.append(0)
            
df['scene_values'] = scene_values


street_values = []
for i, name in enumerate(T['Улица']):
    try:
        if str(name) != 'nan':
            street_values.append(dict_street[name])
        else:
            street_values.append(0)
    except:
        street_values.append(0)
df['street_values'] = street_values

    
house_values = []
for i, name in enumerate(T['Дом']):
    try:
        if str(name) != 'nan':
            house_values.append(dict_house[name])
        else:
            house_values.append(0)
    except:
        house_values.append(0)
    
df['house_values'] = house_values

    
road_values = []
for i, name in enumerate(T['Дорога']):
    try:
        if str(name) != 'nan':
            road_values.append(dict_road[name])
        else:
            road_values.append(0)
    except:
        road_values.append(0)
df['road_values'] = road_values



km_values = []
for i, name in enumerate(T['Километр']):
    try:
        if str(name) != 'nan':
            km_values.append(dict_km[name])
        else:
            km_values.append(0)
    except:
        km_values.append(0)
df['km_values'] = km_values


meter_values = []
for i, name in enumerate(T['Метр']):
    try:
        if str(name) != 'nan':
            meter_values.append(dict_meter[name])
        else:
            meter_values.append(0)
    except:
        meter_values.append(0)
df['meter_values'] = meter_values




# # Предсказание
arr = clf.predict(df)

file = pd.DataFrame()
file['id'] = T['id']
file['Погибло'] = [arr[i][0] for i in range(len(arr))]
file['Погибло детей'] = [arr[i][1] for i in range(len(arr))]
file['Ранено'] = [arr[i][2] for i in range(len(arr))]
file['Ранено детей'] = [arr[i][3] for i in range(len(arr))]

file.to_csv(args.out_path_to_csv_file, index=False)
print(f'Предсказания сохранены в файле {args.out_path_to_csv_file}')

# file = pd.read_csv(args.out_path_to_csv_file)
# print(sum(file['Погибло']), sum(file['Погибло детей']), sum(file['Ранено']), sum(file['Ранено детей']))
