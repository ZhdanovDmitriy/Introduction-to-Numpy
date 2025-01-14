import numpy as np
import array
import sys

#Типы данных в Python меняется в RunTime(динамическая типизация)
# x = 1
# print(type(x))
# print(sys.getsizeof(x))#28 bites
# x = 'hello'
# print(type(x))
# print(sys.getsizeof(x))#54 bites
# x = True
# print(type(x))
# print(sys.getsizeof(x))#28 bites

# l1 = list([])
# print(sys.getsizeof(l1))#56 bites

# l2 = list([1, 2, 3])
# print(sys.getsizeof(l2))#88 bites

# l3 = list([1, "str", True])
# print(sys.getsizeof(l3))#88 bites

#В работе с большими данными критически важно избегать
#автотипизацию
#В питоне есть аналог - array(массив), хранящий только
#один тип данных

# a1 = array.array('i', [1,2,3])
# print(type(a1))
# print(sys.getsizeof(a1))#92 bites

## 1. Коды типов в python - аналог в C/C++ -> тип в Python:
# b	- signed char -> int
# B	- unsigned char	-> int
# h	- signed short	-> int
# H	- unsigned short -> int
# i	- signed int -> int
# I	- unsigned int -> int
# l	- signed long -> int
# L	- unsigned long -> int
# q	- signed long long -> int
# Q	- unsigned long long -> int
# f	- float -> float
# d - double -> float

# ## 2. Код с другим типом:
# a2 = array.array('b', [1,2,3])
# print(type(a2))
# print(sys.getsizeof(a2))#83 bites

# a3 = array.array('d', [1.0000001,2.2,3])
# print(type(a3))
# print(sys.getsizeof(a3))#104 bites

#numpy ориентирован на эфеффективное 
#хранение и взаимодействие с данными

# a = np.array([1,2,3,4,5])
# print(type(a), a)

# a = np.array([1,2.2,3,4,5]) #автоматическое повышающее приведение типов
# print(type(a), a)
# a = np.array([1,2.2,3,4,5], dtype = int) #конкретное приведение
# print(type(a), a)

# #генерация массивов
# a = np.array([range(i, i+3) for i in [2,4,6]])
# print(type(a), a)
# a = np.zeros(10, dtype = int)
# print(a, type(a))
# print(np.ones((3,5), dtype=float))
# print(np.full((4,5), 3.1415))
# print(np.arange(0,20,2))
# print(np.eye(4))#квадратная единичная матрица

# # 4. #Напишите код для создания массива с 5 равномерно распределенными случайными значениями в диапазоне от 0 до 1
# print(np.arange(1/6, 1-1/6, 1/6))

# ## 5. Напишите код для создания массива с 5 нормально распределенными случайными значениями с мат. ожиданием = 0 и дисперсией 1
# mu = 0 #матиматическое ожидание
# #дисперсия D: D^2 = sigma => sigma = 1
# sigma = 1 #стандартное отклонение
# print(np.random.normal(mu, sigma, 5))

# ## 6. Напишите код для создания массива с 5 случайными целыми числами в от [0, 10)
# print(np.random.randint(10, size = 5))


## МАССИВЫ
np.random.seed(1) #псевдослучайные числа

# x1 = np.random.randint(10, size = 3)
# x2 = np.random.randint(10, size = (3,2))
# x3 = np.random.randint(10, size = (3,2,2))
# print(x1, "\n")
# print(x2, "\n")
# print(x3,"\n")

#Свойства(Атрибуты) массивов:
#1)Число размерностей(.dnim)
#2)Размер каждой размерности (.shape)
#3)Общий размер массива(.size)

# print(x1.ndim, x1.shape, x1.size)
# print(x2.ndim, x2.shape, x2.size)
# print(x3.ndim, x3.shape, x3.size)

#Доступ к элементам массива:
#Индексы(с 0)
# a = np.array([1,2,3,4,5])
# print(a[0])
# print(a[-2])
# a[1] = 20
# print(a)
# a = np.array([[1,2],[3,4]])
# print(a[0][0])
# print(a[-1][-1])
# a[1,0] = 100
# print(a)

# a = np.array([1, 2, 3, 4])
# b = np.array([1.0, 2, 3, 4])
# print(a)
# print(b)
# a[0] = 10
# print(a)
# a[0] = 10.123 #Изменение типов не происходит
# print(a)

##Срез массива [start = 0, finish = shape, step = 1]
# a = np.array([1,2,3,4,5,6])
# print(a[0:3:1])
# print(a[:3])
# print(a[3:])
# print(a[1:5])
# print(a[1:-1])
# print(a[1::2])
# print(a[::-1]) #step = -1 меняет местами параметры start и finish


## 7. Написать код для создания срезов массива 3 на 4
## - первые две строки и три столбца
## - первые три строки и второй столбец
## - все строки и столбцы в обратном порядке
## - второй столбец
## - третья строка
#a = np.random.randint(0,10, size = (3,4))
#print(a, "\n")
#print(a[:2, :3], "\n")
#print(a[:3, 1:2:], "\n")
#print(a[::-1, ::-1], "\n")
#print(a[::, 1:2:], "\n")
#print(a[2:3:], "\n")

#Срез это не копия, это ссылка на исходный массив
# a = np.array([1,2,3,4,5,6])
# b = a[:3]
# print(b)
# b[0] = 100
# print(a)

## 8. Продемонстрируйте, как сделать срез-копию
# a = np.array([1,2,3,4,5,6])
# b = np.copy(a[:3])
# print(b)
# b[0] = 100
# print(b)
# print(a)


# a = np.arange(1,13)
# print(a)
# print(a.reshape(2,6))#Позволяет вектор переводить в многомерный массив

## 9. Продемонстрируйте использование newaxis для получения вектора-столбца и вектора-строки
# a = np.array([1, 2, 3, 4])
# print(a)

# column_vector = a[:, np.newaxis]
# print(column_vector)
# print(column_vector.shape)  # (4, 1)

# row_vector = a[np.newaxis, :]
# print(row_vector)
# print(row_vector.shape)  # (1, 4)

# x = np.array([1,2,3])
# y = np.array([4,5])
# z = np.array([6])
# print(np.concatenate([x,y,z]))

# x = np.array([1,2,3])
# y = np.array([4,5,6])
# r1 = np.vstack([x,y]) #вертикальная склейка
# print(r1)
# print(np.hstack([r1,r1])) #горизонтальная склейка



## 10. Разберитесь, как работает метод dstack
# Метод numpy.dstack используется для объединения массивов вдоль новой оси.(один массив за другой) 
# Это полезно, когда нужно объединить, например, двумерные массивы, чтобы получить трёхмерный массив, 
# где каждый исходный массив становится "слоем".
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6], [7, 8]])
# result = np.dstack((a, b))
# print(result)
# print(result.shape)

## 11. Разберитесь, как работают методы split, vsplit, hsplit, dsplit

# # Разбивает массив на несколько подмассивов вдоль заданной оси
# # numpy.split(array, indices_or_sections, axis=0) 
# # (массив, кол-во частей, ось)
# a = np.array([1, 2, 3, 4, 5, 6])
# result = np.split(a, 3)
# print(result)

# #Разбивает массив вдоль вертикальной оси (по строкам).
# #numpy.vsplit(array, indices_or_sections)
# #(массив, кол-во частей)
# a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# result = np.vsplit(a, 2)
# print(result)

# #Разбивает массив вдоль горизонтальной оси (по столбцам).
# #numpy.hsplit(array, indices_or_sections)
# #(массив, кол-во частей)
# a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# result = np.hsplit(a, 2)
# print(result)

# #Разбивает массив вдоль оси глубины (третьей оси, axis = 2).
# #numpy.dsplit(array, indices_or_sections)
# #(массив, кол-во частей)
# a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# result = np.dsplit(a, 2) 
# print(result)

###Вычисления с массивами

#Векторизиованная операция - операция, применяемая к каждому
#элементу массива

# x = np.arange(10)
# print(x)

# print(x*2 + 1)
# #Универсальные функции делают то же самое

# print(np.add(np.multiply(x,2),1))

# # - - / // ** %

# ## 12. Привести пример использования всех универсальных функций, которые я привел
# print(np.subtract(x,1))
# print(np.negative(x))
# print(np.divide(x,2))
# print(np.floor_divide(x,2))
# print(np.pow(x,2))
# print(np.divmod(x,2))

## есть еще np.abs, sin/cos/tan/arctan, exp, log и т.д

# x = np.arange(5)
# y = np.zeros(10)
# print(np.multiply(x,10, out=y[::2]))
# print(y)

# x = np.arange(1,5)
# print(x)
# #reduce - свертка к одному элементу
# print(np.add.reduce(x))
# #accumulate - свертка к одному элементу по шагам
# print(np.add.accumulate(x))

#Векторные произведения
x = np.arange(1,10)
#outer - для всех пар
print(np.add.outer(x,x)) #таблица сложения
print(np.multiply.outer(x,x)) #таблица умножения