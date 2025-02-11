import numpy as np
import pandas as pd

#1. Сценарий
#2. Командная оболочка - IPython
#3. Jupyter

# 1 Сценарий
# plt.show() - запускается только 1 раз
# Логика следующая: код создает объекты класса figure, 
# которые отображаются в зависимости от условия

import matplotlib.pyplot as plt
# x = np.linspace(0,10,100)
# fig = plt.figure()
# plt.plot(x, np.sin(x))
# plt.show()
#plt.plot(x, np.cos(x))#уже не нарисуется

#2. Ipython
# %matplotlib
# import matplotlib.pyplot as plt
# plt.plot - любой вызов данный команды будет сразу открывать
# окно графика => plt.show() не нужно
# Может залагать, тогда добавляют plt.draw()
# Если не нужна консольная информация, то в конце функции
# надо написать ;

# Jupyter
# %matplotlib inline - в блокнот добавляется статическая картинка
# %matplotlib notebook - в блокнот добавляются интерактивные графики


#Возможность сохранения графиков в файлы:
# fig.savefig('saved_images.png')

#print(fig.canvas.get_supported_filetypes()) - список доступных форматов


# Два способа вывода графиков:
# 1) MATLAB-подобный стиль
# 2) в ООП стиле

# 1. Матлаб - подобный стиль(минусы - сложно возращаться к старым графикам)
x = np.linspace(0,10,100)

# plt.figure()
# plt.subplot(2,1,1)#число строк, число колонок, номер
# plt.plot(x, np.sin(x))
# plt.subplot(2,1,2)
# plt.plot(x, np.cos(x))
# plt.show()

# 2. ООП стиль (плюсы - позволяет вернуться к предыдущему графику)
# fig:plt.Figure - контейнер, содержащий все объекты(СК, тексты, метки)
# ax:Axes - система координат - прямоугольник, деления, метки

# fig, ax = plt.subplots(2) #subplot из двух элементов
# ax[0].plot(x, np.sin(x))
# ax[1].plot(x, np.cos(x))
# ax[0].plot(x, np.exp(x))
# plt.show()

#Цвета линии color
# - через готовые цвета 'blue'
# rgb/cmyk -> 'rg'
# '0.14' - градация серого [0,1]
# RGB - (1.0, 0.2, 0.3)
# HTML - 'salmon'

#Стили линий linestyle:
# - сплошная '-', 'solid'
# - штриховая '--', 'dashed'
# - штрих-пунктирная '-.-', 'dashdot'
# - пунктирная ':', 'dotted'

# fig = plt.figure()
# ax = plt.axes()
# ax.plot(x, np.cos(x), color = 'blue', linestyle = 'solid')
# ax.plot(x, np.sin(x-1), color = 'g', linestyle = 'dashed')
# ax.plot(x, np.sin(x-2), color = '0.75', linestyle = 'dashdot')
# ax.plot(x, np.sin(x-3), color = '#FF00EE')
# ax.plot(x, np.sin(x-4), color = (1.0, 0.2, 0.3))
# ax.plot(x, np.sin(x-5), color = 'salmon')
# ax.plot(x, np.sin(x-6), '--k')

# fix, ax = plt.subplots(4)
# ax[0].plot(x, np.sin(x))
# ax[1].plot(x, np.sin(x))
# ax[2].plot(x, np.sin(x))
# ax[3].plot(x, np.sin(x))

# ax[1].set_xlim(-2, 12) #Установка пределов по x
# ax[1].set_ylim(-1.5, 1.5) #Установка пределов по y

# #Отзеркаливание графика
# ax[2].set_xlim(12, -2) #Установка пределов
# ax[2].set_ylim(1.5, -1.5) #Установка пределов

# ax[3].autoscale(tight=True)


# plt.subplot(3,1,1)
# plt.plot(x, np.sin(x))

# plt.title("Синус")
# plt.xlabel("x")
# plt.ylabel("sin(x)")


# plt.subplot(3,1,2)
# plt.plot(x, np.sin(x), '-g', label='sin(x)')
# plt.plot(x, np.cos(x), ':b', label='cos(x)')

# plt.title("Синус и косинус")
# plt.xlabel("x")
# plt.legend()

# plt.subplot(3,1,3)
# plt.plot(x, np.sin(x), '-g', label='sin(x)')
# plt.plot(x, np.cos(x), ':b', label='cos(x)')

# plt.title("Синус и косинус")
# plt.xlabel("x")
# plt.axis('equal')

# plt.subplots_adjust(hspace=0.5)

# x = np.linspace(0,10,30)
# plt.plot(x, np.sin(x), 'o', color = 'g')
# plt.plot(x, np.sin(x)+1, '>', color = 'g')
# plt.plot(x, np.sin(x)+2, '^', color = 'g')
# plt.plot(x, np.sin(x)+3, 's', color = 'g')

# plt.plot(x, np.sin(x), '--p', markersize=15, linewidth = 10,
#         markerfacecolor = 'white', markeredgecolor = 'gray',
#         markeredgewidth =2 ) #Тонкая настройка маркеров

# rng = np.random.default_rng(0)
# colors = rng.random(30)
# sizes = 30 * rng.random(30)
# plt.scatter(x, np.sin(x), marker = 'o', c = colors, s = sizes)#Каждой точке можно задать свои характеристики
# plt.colorbar()

#Plot производительнее чем scatter на больших данных



#Визуализация погрешности
# x = np.linspace(0,10,50)

# dy = 0.4
# y = np.sin(x) + dy * np.random.rand(50)

# plt.errorbar(x,y, yerr = dy, fmt = '.k')
# plt.fill_between(x,y-dy,y+dy, color = 'red', alpha = 0.4)


def f(x,y):
    return np.sin(x) ** 5 + np.cos(20 + x * y) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x,y)
Z = f(X,Y)

# plt.contour(X,Y,Z, cmap = 'RdGy')
plt.contourf(X,Y,Z, cmap = 'RdGy')
plt.colorbar()
plt.show()