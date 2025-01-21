import numpy as np
#суммирование значений

# rng = np.random.default_rng(1)
# s = rng.random(50)
# #равносильно s = np.random.default_rng(1).random(50)

# print(s)
# print(sum(s))
# print(np.sum(s)) #быстрее на больших данных, а так же
# # считает суммы многомерных массивов

# a = np.array([
#     [1,2,3,4,5],
#     [6,7,8,9,10]
# ])

# print(np.sum(a))
# print(np.sum(a, axis = 0)) 
# print(np.sum(a, axis = 1)) 
# #по какому измерению будем свертывать? 0 - строки, 1 - столбцы
# print(np.min(a, axis = 0)) 
# print(np.max(a, axis = 1)) 
# #Эквивалентно
# print(a.min(0)) 
# print(a.min(1)) 

# #Префикс nan важен в ML - это безопасная версия
# #запуска функция, разрешающие наличие NaN в значениях
# print(np.nanmin(a))
# print(np.nanmin(a, axis = 0)) 
# print(np.nanmax(a, axis = 1)) 


#ТРАНСЛИРОВАНИЕ(broadcasting)
#набор правил, позволяющий осуществлять бинарные операции
#с массивами разных форм и размеров

# a = np.array([0,1,2])
# b = np.array([5,5,5])

# print(a + b)
# print(a + 5) #скаляр транслируется в массив(подстраивается под размер)

# a = np.array([[0,1,2], [3,4,5]])
# print(a+5)

# a = np.array([0,1,2])
# b = np.array([[0],[1],[2]])
# print(a + b)
# print(b + a)

#Правила транслирования
# 1) Если размерности массивов отличаются, то
# форма массива с меньшей размерностью дополняется 1 с
# левой стороны
# 2)Если формсы массивов не совпадают в каком-то измерении,
# то если у массива форма равна 1, то он растягивается до
# соответствия формы второго
# 3)Если после применения этого правила в каком-то из
# измерений размеры отличаются и ни один из них не равен 1,
# то генерируется ошибка


# a = np.array([[0,1,2], [3,4,5]])
# b = np.array([5])
# print(a.ndim, a.shape)
# print(b.ndim, b.shape)
# print(a + b)

# # a         (2,3)
# # b(1,) -> b(1,1) -> (2,3)


#Рабочий пример:
# a = np.ones((2,3))
# b = np.arange(3)

# print(a, a.ndim, a.shape)
# print(b, b.ndim, b.shape)
# # (2,3) -{1 step}-> (2,3) -{2 step}-> (2,3)
# # (3,)  -{1 step}-> (1,3) -{2 step}-> (2,3)
# c = np.array([[1,2,3]])
# print(c.shape)
# c = np.array([[1,2,3], [1,2,3]])
# print(c.shape)
# c = a + b
# print(c, c.shape)

# Другой работающий пример
# a = np.arange(3).reshape((3,1))
# b = np.arange(3)
# print(a)
# print(b)

# # (3,1) -{1 step}-> (3,1) -{2 step}-> (3,3)
# # (3,)  -{1 step}-> (1,3) -{2 step}-> (3,3)

# c = a + b
# # [ 0 0 0 ]         [ 0 1 2 ]
# # [ 1 1 1 ]    +    [ 0 1 2 ]
# # [ 2 2 2 ]         [ 0 1 2 ]
# print(c, c.shape)

# Пример с ошибкой
# a = np.ones((3,2))
# b = np.arange(3)

# # (3,2) -{1 step}-> (3,2) -{2 step}-> (3,2) -{3 step}-> Error
# # (3,)  -{1 step}-> (1,3) -{2 step}-> (3,3) -{3 step}-> Error
# # Мы не знаем, как растянуть размерность 2, знаем только 1

# X = np.array([
#     [1,2,3,4,5,6,7,8,9],
#     [9,8,7,6,5,4,3,2,1]
# ])
# Xmean0 = X.mean(0)
# print(Xmean0)
# Xcenter0 = X - Xmean0
# print(Xcenter0)#Отцентровали массив по 0

# Xmean1 = X.mean(1)
# print(Xmean1)
# Xmean1 = Xmean1[:, np.newaxis]#Повернули(превратили в столбец)
# Xcenter1 = X - Xmean1
# print(Xcenter1)#Отцентровали массив

# x = np.linspace(0,5,50)
# y = np.linspace(0,5,50)[:, np.newaxis]
# z = np.sin(x)**3 + np.cos(20+y*x) * np.sin(y)
# print(z.shape)

# #можно нарисовать
# import matplotlib.pyplot as plt
# plt.imshow(z)
# plt.colorbar()
# plt.show()

#Условия
# x = np.array([1,2,3,4,5])
# y = np.array([[1,2,3,4,5], [6,7,8,9,10]])
# print(x<3)#универсальная функция
# print(np.less(x,3))

# print(np.sum(x < 3)) # количество элементов меньше 3
# print(np.sum(y<4)) #все эелемнты меньше 4
# print(np.sum(y<4, axis = 0)) #свертка по столбцам
# print(np.sum(y<4, axis = 1)) #свертка по строкам

# print(x[x<3])

# # Есть побитовые операции 

# # Векторизация индекса
# x = np.array([0,1,2,3,4,5,6,7,8,9])
# index = [1,5,7]
# print(x[index])

# index = [[1,5,7],[2,4,8]]
# print(x[index])
#Форма результата отражает форму массива индексов, а 
#не форму исходного массива

# x = np.arange(12).reshape((3,4))

# print(x)
# print(x[2])
# print(x[2, [2,0,1]])
# print(x[1:, [2,0,1]])

# x = np.arange(10)
# i = np.array([2,1,8,4])
# print(x)
# x[i] = 999
# print(x)

# #Сортировка массивов
# x = [3,2,3,5,2,6,7,3,6,3,2]

# print(sorted(x))
# print(np.sotr(x)) #лучше на больших объемах данных

#Структурированные массивы
data = np.zeros(4, dtype = {
    'names':(
        'name', 'age'
    ),
    'formats':(
        'U10', 'i4' #строка Юникод 10 байт, и целое 4 байта
    )
})
print(data.dtype)

name = ['name1','name2','name3','name4']
age = [10, 20, 30, 40]
data['name'] = name;
data['age'] = age;

print(data)
print(data[data['age'] > 20]['name'])

#Массивы записей
data_rec = data.view(np.recarray)

print(data_rec)
print(data_rec[0])
print(data_rec[-1].name)