import numpy as np
import pandas as pd
# # 1. Разобраться как использовать мультииндексные ключи в данном примере
index = [
    ('city_1', 2010),
    ('city_1', 2020),
    ('city_2', 2010),
    ('city_2', 2020),
    ('city_3', 2010),
    ('city_3', 2020),
]

population = [
    101,
    201,
    102,
    202,
    103,
    203,
]
pop = pd.Series(population, index=pd.MultiIndex.from_tuples(index))#создаем мультиидексы
pop_df = pd.DataFrame(
    {
        'total': pop,
        'something': [
            10,
            11,
            12,
            13,
            14,
            15,
        ]
    }
)
print(pop_df)
print(pop_df.loc['city_1', 'something'])
print(pop_df.loc[['city_1', 'city_3'], ['total', 'something']])
print(pop_df.loc[['city_1', 'city_3'], 'something'])

# ???? ## pop_df_1 = pop_df.loc???['city_1', 'something']
# ???? ## pop_df_1 = pop_df.loc???[['city_1', 'city_3'], ['total', 'something']]
# ???? ## pop_df_1 = pop_df.loc???[['city_1', 'city_3'], 'something']


print('===================')
# 2. Из получившихся данных выбрать данные по 
# - 2020 году (для всех столбцов)
# - job_1 (для всех строк)
# - для city_1 и job_2 

index = pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010, 2020]
    ],
    names=['city', 'year']
)

columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2']
    ],
    names=['worker','job']
)
rng = np.random.default_rng(1)
data = rng.random((4,6))

data_df = pd.DataFrame(data, index = index, columns = columns)
print(data_df)
print(data_df.loc[(slice(None), 2010), :])
#print(data_df[data_df.index.get_level_values('year') == 2010]) #Альтернатива
print(data_df.loc[:,(slice(None),'job_1')])
print(data_df.loc['city_1', ((slice(None),'job_2'))])

print('=================')
# 3. Взять за основу DataFrame со следующей структурой
index = pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010, 2020]
    ],
    names=['city', 'year']
)
columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2']
    ],
    names=['worker', 'job']
)
# Выполнить запрос на получение следующих данных
# - все данные по person_1 и person_3
# - все данные по первому городу и первым двум person-ам (с использование срезов)

rng = np.random.default_rng(1)
data = rng.random((4,6))
data_df = pd.DataFrame(data, index = index, columns = columns)
print(data_df)
print(data_df.loc[:, ['person_1','person_3']])
print(data_df.loc[('city_1',slice(None)), 'person_1':'person_2'])

# Приведите пример (самостоятельно) с использованием pd.IndexSlice
print(data_df.loc[pd.IndexSlice[:, 2010], :]) #выбор данных за 2010 год
print(data_df.loc[pd.IndexSlice['city_1', 2010], pd.IndexSlice[:, 'job_1']]) #Выбор данных для 'city_1' за 2010 год для job_1



#4. Привести пример использования inner и outer джойнов для Series (данные примера скорее всего нужно изменить)
ser1 = pd.Series(['a', 'b', 'c'], index=[1, 2, 3])
ser2 = pd.Series(['b', 'c', 'f'], index=[4, 2, 1])

print(pd.concat([ser1, ser2]))
print(pd.concat([ser1, ser2], axis=1, join="outer"))#берутся все уникальные индексы, если в каких-то сериях нет данных для какого-то индекса,
# то ставится NaN 
print(pd.concat([ser1, ser2], axis=1, join="inner"))#берутся совпадающие индексы и соответствующие им данные
