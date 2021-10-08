import numpy as np
from mayavi.mlab import *
import pickle
with open('1a28_grid.pkl','rb') as f:
    grid = pickle.load(f)[0]
with open('1a28_heatmap.pkl', 'rb') as f:
    heatmap = pickle.load(f)
    
x = []
y = []
z = []
c = []
for i in range(20):
    for j in range(20):
        for k in range(20):
            tmp = grid[i,j,k]
            if tmp[15:].sum() >= 1:
                x.append(i)
                y.append(j)
                z.append(k)
                n = list(tmp).index(1.0)
                if n in [15,16,17]:
                    c.append(1)
                elif n in [18,19,20]:
                    c.append(1.1)
                elif n == 21:
                    c.append(1.2)
                else:
                    c.append(0.9)
            elif tmp[1:14].sum() >= 1:
                x.append(i)
                y.append(j)
                z.append(k)
                c.append(0.3)

        
figure(bgcolor=(1, 1, 1))
contour3d(heatmap, opacity = 0.3,colormap='cool')
points3d(x, y, z, c,scale_factor=.5,colormap='blue-red')
colorbar()
show()
