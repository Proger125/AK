import matplotlib.pyplot as plt

f = open('output.txt')
x_list = []
y_list = []
z_list = []
for line in f:
    point = line.split()
    x_list.append(float(point[0]))
    y_list.append(float(point[1]))
    z_list.append(float(point[2]))

ax = plt.axes(projection='3d')
ax.plot_trisurf(x_list, y_list, z_list, cmap='binary')
plt.show()
