import pandas as pd
import numpy as np
# 1. Привести различные способы создания объектов типа Series
# Для создания Series можно использовать
# - списки Python или массивы NumPy
# - скалярные значение
# - словари
data = pd.Series([1, 2, 3, 4, 5])
print(data)
data = pd.Series(0, index=[0, 1, 2, 3, 4])
print(data)
data = pd.Series(
    {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5
    }
)
print(data)

# 2. Привести различные способы создания объектов типа DataFrame
# DataFrame. Способы создания
# - через объекты Series
# - списки словарей
# - словари объектов Series
# - двумерный массив NumPy
# - структурированный массив Numpy

data1 = pd.Series([1, 2, 3])
data2 = pd.Series([3, 2, 1])
df = pd.DataFrame([data1, data2])
print(df)

data = [
    {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5},
    {'a': 6, 'b': 7, 'c': 8}
]
df = pd.DataFrame(data, dtype = 'Int32')
print(df)

df = pd.DataFrame({
    'first' : data1,
    'second' : data2
})
print(df)

df = pd.DataFrame(np.random.default_rng().integers(0,10,(5,5)))
print(df)

data = np.array([(1.003, 'one'), (1.98764, 'two'), (3.00103, 'three')], dtype=[('num', 'float64'), ('word', 'U10')])
df = pd.DataFrame(data)
print(df)

# 3. Объедините два объекта Series с неодинаковыми множествами ключей (индексов) так, чтобы вместо NaN было установлено значение 1
population_dict = {
    "city_1": 1001,
    "city_2": 1002,
    "city_3": 1003,
    "city_41": 1004,
    "city_51": 1005,
}
area_dict = {
    "city_1": 9991,
    "city_2": 9992,
    "city_3": 9993,
    "city_42": 9994,
    "city_52": 9995,
}
population = pd.Series(population_dict, dtype = 'Int32')
area = pd.Series(area_dict, dtype = 'Int32')
states = pd.DataFrame({
    "population": population,
    "area": area,
})
states = states.fillna(1)
print(states)

# 4. Переписать пример с транслирование для DataFrame так, чтобы вычитание происходило по СТОЛБЦАМ
rng = np.random.default_rng()
A = rng.integers(0, 10, (5, 5))
print(A)
df = pd.DataFrame(A, columns=['a', 'b', 'c', 'd', 'e'])
print(df)
print(df.iloc[:, 4])
print(df.sub(df.iloc[:, 4], axis=0))

# 5. На примере объектов DataFrame продемонстрируйте использование методов ffill() и bfill()
data = {'a': [1, np.nan, 3, np.nan],
        'b': [np.nan, 2, np.nan, 4]}
df = pd.DataFrame(data)
print(df)
df = df.ffill()
print(df)

data = {'a': [1, np.nan, 3, np.nan],
        'b': [np.nan, 2, np.nan, 4]}
df = pd.DataFrame(data)
df = df.bfill()
print(df)