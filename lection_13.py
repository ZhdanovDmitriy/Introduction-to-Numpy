#Метод опорных векторов (SCM - support vector machine) - классификация и регрессия
#Разделяющая классификация
#Выбирается линия с максимальным отступом
#Этот метод, в отличие от прошлой пары - разделяющая классификация (рисует линию/кривую, которая разделяет классы данных)

# Достоинства
# - Зависимость от небольшого числа опорных векторов => компактность модели
# - После обучения предсказания проходит очень быстро
# - На работу методов влияют только точки, находящиеся возле отступов, поэтому метдоы подходят для многомерных данных

# Недостатки
# - При большом количестве обучающих образцов могут быть значительные вычислительные затраты
# - Большая зависимость от С - размытости. Поиск параметра может привести к большим вычислительным затратам.
# - У результатов отсутствует вероятностная интерпретация (точки могут не участвовать в принятии решения)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

iris = sns.load_dataset('iris')

'''
print(iris.head())

data = iris[["sepal_length", "petal_length", "species"]]
data_df = data[(data["species"] == "setosa") | (data["species"] == "versicolor")]

X = data_df[["sepal_length", "petal_length"]]
y = data_df["species"]

data_df_seposa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

plt.scatter(data_df_seposa["sepal_length"], data_df_seposa["petal_length"])
plt.scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

model =  SVC(kernel = 'linear', C=10000)
model.fit(X, y)

print(model.support_vectors_)

plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s = 400,
    facecolor='none',
    edgecolor='black'
) # Получили опорные точки

x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]),100)
x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]),100)
X1_p, X2_p = np.meshgrid(x1_p, x2_p)
X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"])
y_p = model.predict(X_p)
X_p["species"] = y_p
X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]
plt.scatter(X_p_setosa["sepal_length"], X_p_setosa["petal_length"], alpha = 0.2)
plt.scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha = 0.2)
'''
'''
data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']


c_value = [[10000,1000,100,10], [1, 0.1, 0.01, 0.001]]
fig, ax = plt.subplots(2,4, sharex='col', sharey='row')
for i in range(2):
    for j in range(4):
        ax[i,j].scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
        ax[i,j].scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

        # Если С большое, то отступ задается "жестко", нет размытия. Чем меньше С, тем отступ становится более размытым.
 
        model = SVC(kernel='linear', C=c_value[i][j])
        model.fit(X,y)

        #print(model.support_vectors_)

        ax[i,j].scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=400, facecolor="none", edgecolors='black')


        x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
        x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

        X1_p, X2_p = np.meshgrid(x1_p, x2_p)

        X_p = pd.DataFrame(
            np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
        )

        y_p = model.predict(X_p)

        X_p["species"] = y_p

        X_p_virginica = X_p[X_p['species'] == 'virginica']
        X_p_versicolor = X_p[X_p['species'] == 'versicolor']

        ax[i,j].scatter(X_p_virginica['sepal_length'], X_p_virginica['petal_length'], alpha=0.1)
        ax[i,j].scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.1)

plt.show()
'''


# Деревья решений и случайны леса
# Случ леса - непараметрический алгоритм (тк в формуле модели-обучающей функции -непосредственно данные не заложены. В параметрических фиксируем формулу заранее, как лин регрессия, потом подбир параметры, а тут мы не знаем, какие параметры возьмем, все зависит от данных)
# СЛ - пример ансамблевого метода, основанного на агрегации результатов множества простых моделей.
# основан на дереве принятия решений
# В реализациях дерева принятия решений в машинном обучении, вопросы обычно ведут к разделению данных по осям, т.е. каждый узел разбивает данные на 2 группы по одному из признаков.

from sklearn.tree import DecisionTreeClassifier

species_int = []
for r in iris.values:
    match r[4]:
        case "setosa":
            species_int.append(1)
        case "versicolor":
            species_int.append(2)
        case "virginica":
            species_int.append(3)
            
species_int_df = pd.DataFrame(species_int)
print(species_int_df.head())

data = iris[['sepal_length', 'petal_length']]
data["species"] = species_int_df

print(data)

data_df = data[(data['species'] == 3) | (data['species'] == 2)]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 3]
data_df_versicolor = data_df[data_df['species'] == 2]


max_depth = [[1,2,3,4], [5,6,7,8]]
fig, ax = plt.subplots(2,4, sharex='col', sharey='row')

for i in range(2):
    for j in range(4):
        ax[i,j].scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
        ax[i,j].scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])
        model = DecisionTreeClassifier(max_depth=max_depth[i][j])
        model.fit(X,y)

        
        x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
        x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

        X1_p, X2_p = np.meshgrid(x1_p, x2_p)

        X_p = pd.DataFrame(
            np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
        )
        y_p = model.predict(X_p)

        ax[i,j].contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.4, levels=2, cmap='rainbow', zorder = 1)
plt.show()

'''plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])
model = DecisionTreeClassifier(max_depth=5)
model.fit(X,y)
'''
#plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.2)
#plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.2)

'''
plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

'''
plt.show()
