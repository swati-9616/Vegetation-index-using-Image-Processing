# -*- coding: utf-8 -*-

from function_list import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n = input("Order of IDW :")
k= input("No of grids :")
n = int(n)
k = int(k)
df = pd.read_excel('DataFile.xlsx')

x = np.array(df['X'])
y = np.array(df['Y'])
z = np.array(df['Z'])

# generating random data for test
# x = np.array([3,2,3,1,6,7])
# y = np.array([1,1,3,4,7,8])
# z = np.array([4,1,5,7,8,10])

# Finding minimum and maximum of x-y coordinates to create empty grid
x_min = min(x)
x_max = max(x)
y_min, y_max = min(y), max(y)

# No of square grid

d = (x_max-x_min)/k

# Creating range for x-coordinate and y-coordinate
X = np.arange(x_min, x_max+1, d)
x_len = len(X)
Y = np.arange(y_min, y_max+1, d)
y_len = len(Y)

# creating mesh grid
x_grid, y_grid = np.meshgrid(X,Y)
# Converting x and y grid into single array
x_grid = x_grid.flatten()
y_grid = y_grid.flatten()

# Calculating distance for each grid point from known point
D = list(map(lambda i: distance(x_grid[i],y_grid[i],x,y),np.arange(0,len(x_grid))))

# Estimating the value using IDW interpolation
Z = list(map(lambda i: IWD(z,D[i],n),np.arange(0,len(x_grid))))
# Converting 1D array of Z into desired grid frame
Z = np.array(Z)
Z = Z.reshape(y_len, x_len)

# Plotting the result
x_grid, y_grid = np.meshgrid(X,Y)
fig = plt.figure()
ax = plt.gca(projection = '3d')
surf = ax.plot_surface(x_grid, y_grid, Z, cmap='terrain')
fig.colorbar(surf, ax = ax,
             shrink = 0.5, aspect = 5)
plt.title('IDW Interpolation of order ' + str(n))
