import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

###График 1
x1 = np.linspace(0,20,5)
y1 = 12*np.random.random(5)
y2 = 12*np.random.random(5)
plt.figure()
plt.plot(x1,y1, 'r', marker='o', label = 'line1')
plt.plot(x1,y2, 'g--', marker='o', label = 'line2')
plt.legend(loc='upper left')

###График 2
x1 = np.linspace(0,5,5)
y1 = 12*np.random.random(5)
y2 = 12*np.random.random(5)
plt.figure()
grid = plt.GridSpec(2,2)
plt.subplot(grid[0,:2])
plt.plot(x1,y1)
plt.subplot(grid[1,0])
plt.plot(x1,y2)
plt.subplot(grid[1,1])
plt.plot(x1,4*np.sin(y2))

###График 3
fig, ax = plt.subplots()
x = np.linspace(5,-5, 20)
ax.plot(x, x*x)
ax.annotate('min', xy = (0, 0), xytext = (0,10), arrowprops=dict(facecolor='green'))

###График 4
plt.figure()
x = np.random.uniform(0, 7, 250)
y = np.random.uniform(0, 7, 250)
plt.hist2d(x,y, bins=7)
cb = plt.colorbar()

###График 5
x1 = np.linspace(0,5,100)
y1 = np.cos(x1*np.pi)
y2 = 0
plt.figure()
plt.plot(x1,y1, 'r')
plt.fill_between(x1, y1, y2)

###График 6
x1 = np.linspace(0,5,10000)
y1 = np.cos(x1*np.pi)
y1[y1 < -0.5] = np.nan
plt.figure()
plt.plot(x1,y1,linewidth=3)
plt.xlim(0, 5)
plt.ylim(-1, 1)

###График 7
x = np.linspace(0,6,10000)
y = np.ceil(x)
y2 = x
plt.figure()
plt.subplot(1,3,1)
plt.plot(x,y, 'g')
plt.grid(True)
includes = np.where((np.abs(np.round(x) - x) < 0.001) & (np.abs(np.round(y) - y2) < 0.001))[0]
plt.scatter(x[includes], y[includes], color='green', s=20)

plt.subplot(1,3,2)
y = np.floor(x)
plt.plot(x,y, 'g')
plt.grid(True)
includes = np.where((np.abs(np.round(x) - x) < 0.001) & (np.abs(np.round(y) - y2) < 0.001))[0]
plt.scatter(x[includes], y[includes], color='green', s=20)

plt.subplot(1,3,3)
y = np.round(x)
plt.plot(x,y, 'g')
plt.grid(True)
includes = np.where((np.abs(np.round(x) - x) < 0.001) & (np.abs(np.round(y) - y2) < 0.001))[0]
plt.scatter(x[includes], y[includes], color='green', s=20)


###График 8
x = np.linspace(0,10,10)
y1 =  -0.51 * (x-7) * (x-7) + 25
y2 = -0.6 * (x-5) * (x-5) + 15
y3 = -0.2 * (x-5) * (x-5) + 5
y0 = 0
plt.figure()
plt.plot(x,y1, 'g')
plt.fill_between(x, y1, y0, color='green', label = 'y1')
plt.plot(x,y2, color = 'orange')
plt.fill_between(x, y2, y0, color='orange', label = 'y2')
plt.plot(x,y3, color = 'blue')
plt.fill_between(x, y3, y0, color='blue', label = 'y3')
plt.legend()


###График 9
plt.figure()
cars = ['Ford','Toyota','BMW','AUDI','Jaguar']
RoadAccidents = [20,10,35,15,30]
plt.pie(RoadAccidents, labels=cars, explode=(0,0,0.1,0,0))


###График 10
plt.figure()
cars = ['Ford','Toyota','BMW','AUDI','Jaguar']
RoadAccidents = [20,10,35,15,30]
plt.pie(RoadAccidents, labels=cars, wedgeprops={'width': 0.5})


plt.tight_layout()
plt.show()


