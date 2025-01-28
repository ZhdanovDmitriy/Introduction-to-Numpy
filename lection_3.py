import numpy as np
import pandas as pd
#Pandas - расширение NumPy(структурированные массивы)
#Строки и столбцы идексируются не только числами, но и 
#метками других типов

# Series, DataFrame, Index - три остновных структуры

##Series
# data = pd.Series([0.25, 0.5, 0.75, 1.0])
# print(data)
# print(type(data))
# print(data.values)
# print(type(data.values))
# print(data.index)
# print(type(data.index))

# data = pd.Series([0.25, 0.5, 0.75, 1.0])
# print(data[0])
# print(data[1:3])

# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
# print(data['a'])
# print(data['b':'d'])

# print(type(data.index))

# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = [1, 10.2, 'c', 0])
# print(data[1])
# print(data[10.2:'c'])

# population_dict = {
#     'city_1' : 1001,
#     'city_2' : 1002,
#     'city_3' : 1003,
#     'city_4' : 1004,
#     'city_5' : 1005,
# }

# population = pd.Series(population_dict)
# print(population)
# print(population['city_4':'city_5'])

#Длясоздание Series можно использовать:
# - списки Python или массивы из NumPy
# - скалярные значения
# - словари

##DataFrame - двумерный массив с явно определленными индексами
# Последовательность "согласованных" объектов Series
# population_dict = {
#     'city_1' : 1001,
#     'city_2' : 1002,
#     'city_3' : 1003,
#     'city_4' : 1004,
#     'city_5' : 1005,
# }

# area_dict = {
#     'city_1' : 9991,
#     'city_2' : 9992,
#     'city_3' : 9993,
#     'city_4' : 9994,
#     'city_5' : 9995,
# }

# population = pd.Series(population_dict)
# area = pd.Series(area_dict)

# states = pd.DataFrame({
#     'population' : population,
#     'areal' : area,
# })
# print(states)
# print(states.values)
# print(states.index)
# print(states.columns)

# print(type(states.values))
# print(type(states.index))
# print(type(states.columns))

# print(states['areal'])

#DataFrame. Способы созздания:
# - через объекты Series
# - списки словарей
# - словари объектов Series
# - двумерный массив NumPy
# - структурированный массив NumPy

##Index - способ организации ссылки на данные объектов
# Series и DataFrame.
# Index - неизменяемый, упорядоченный, является мультимножеством(мб одинаковые значения)

# ind = pd.Index([2,3,5,7,11])
# print(ind[1])
# print(ind[::2])
# #ind[1] = 5 #ERROR

# #Index - следует соглашению объекта set(python)
# indA = pd.Index([1,2,3,4,5])
# indB = pd.Index([2,3,4,5,6])
# print(indA.intersection(indB))

# Выюорка данных из Series (похож на словарь)
# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
# print('a' in data)
# print('z' in data)
# print(data.keys())
# print(list(data.items()))
# data['a'] = 100
# data['z'] = 1000
# print(data)

## Можно смотреть как на одномерный массив
# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
# print(data['a':'c']) #Обе границы входят!
# print(data[1:2])
# print(data[(data > 0.5) & (data < 1)])
# print(data[['a','d']])

# #атрибуты - индексаторы
# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = [1, 3, 10, 15])
# print(data.loc[1]) #обращение по заданному индексу
# print(data.iloc[1]) #оьращение по стандартному индексу

#Выбор данных из DataFrame
# pop = pd.Series({
#     'city_1' : 1001,
#     'city_2' : 1002,
#     'city_3' : 1003,
#     'city_4' : 1004,
#     'city_5' : 1005,
# })

# area = pd.Series({
#     'city_1' : 9991,
#     'city_2' : 9992,
#     'city_3' : 9993,
#     'city_4' : 9994,
#     'city_5' : 9995,
# })

# data = pd.DataFrame({
#     'area1' : area,
#     'pop1' : pop,
#     'pop' : pop
# })

# print(data)
# print(data['area1'])
# print(data.area1)

# print(data.pop1 is data['pop1'])
# print(data.pop is data['pop']) #Конфликт

# data['new'] = data['area1']
# data['new1'] = data['area1'] / data['pop1']
# print(data)

#Двумерный NumPy-массив
# pop = pd.Series({
#     'city_1' : 1001,
#     'city_2' : 1002,
#     'city_3' : 1003,
#     'city_4' : 1004,
#     'city_5' : 1005,
# })

# area = pd.Series({
#     'city_1' : 9991,
#     'city_2' : 9992,
#     'city_3' : 9993,
#     'city_4' : 9994,
#     'city_5' : 9995,
# })

# data = pd.DataFrame({
#     'area1' : area,
#     'pop1' : pop,
#     'pop' : pop
# })

# print(data)
# print(data.values)
# print(data.T)
# print(data['area1']) #Обращение к столбцу
# print(data.values[0]) #Обращение к строке

#атрибуты-индексаторы
# print(data.iloc[:3, 1:2])
# print(data.loc[:'city_4', 'pop1':'pop'])
# print(data.loc[data['pop'] > 1002, 'pop1':'pop'])
# print(data.loc[data['pop'] > 1002, ['area1','pop']])
# data.iloc[0,2] = 999999
# print(data)

# Универсальные функции
# rng = np.random.default_rng()
# s = pd.Series(rng.integers(0,10,4))
# print(s)
# print(np.exp(s))

#Различные Series
# pop = pd.Series({
#     'city_1' : 1001,
#     'city_2' : 1002,
#     'city_3' : 1003,
#     'city_41' : 1004,
#     'city_51' : 1005,
# })

# area = pd.Series({
#     'city_1' : 9991,
#     'city_2' : 9992,
#     'city_3' : 9993,
#     'city_42' : 9994,
#     'city_52' : 9995,
# })

# data = pd.DataFrame({
#     'area1' : area,
#     'pop1' : pop
# })
# print(data)

# #Различные DataFrames
# rng = np.random.default_rng()
# dfA = pd.DataFrame(rng.integers(0,10,(2,2)), columns=['a','b'])
# dfB = pd.DataFrame(rng.integers(0,10,(3,3)), columns=['a','b', 'c'])
# print(dfA)
# print(dfB)
# print(dfA + dfB)

rng = np.random.default_rng(1)
A = rng.integers(0, 10,(3,4))
# print(A)
# print(A[0])
# print(A - A[0])#Транслирование

df = pd.DataFrame(A, columns = ['a','b','c','d'])
print(df)
print(df.iloc[0])
print(df - df.iloc[0])
print(df.iloc[0, ::2])
print(df - df.iloc[0, ::2])

#NA - not avalibal value: NaN, null
#Pandas. Два способа хранения отсутствующих значений
#Индикаторы NaN и None
#null

#None - это объект, его использование может привести к накладным рассходам
#Не работает с sum и min
val = np.array([1,2,3])
print(val.sum())

# val = np.array([1,2,None])
# print(val.sum())

val = np.array([1,np.nan, 2,3])
print(np.sum(val))
print(np.nansum(val))

x = pd.Series(range(10), dtype=int)
print(x)
x[0] = None
x[1] = np.nan
print(x)

x1 = pd.Series(['a','b','c'])
print(x1)
x1[0] = None
x1[1] = np.nan
print(x1)


#Специальный элемент NA в Pandas
x2 = pd.Series([1,2,3, np.nan, None, pd.NA])
print(x2)
x3 = pd.Series([1,2,3, np.nan, None, pd.NA], dtype = 'Int32')
print(x3)

print(x3.isnull())
print(x3[x3.notnull()])

print(x3.dropna())

df = pd.DataFrame(
    [
        [1,2,3,None, pd.NA],
        [1,2,3, None,5,6],
        [1,np.nan,3,None,np.nan, 6]
    ]
)
print(df)
# print(df.dropna())
# print(df.dropna(axis = 0))
# print(df.dropna(axis = 1))

#Критерий how выбрасываемости строки и столбца:
#all - отсутствуют все значения
#all - отсутствует хотя бы одно значения
print(df.dropna(axis = 1, how = 'all'))
print(df.dropna(axis = 1, how = 'any'))

#thresh = x, остается, если присутствует минимум x непустых
print(df.dropna(axis=1, thresh=2))