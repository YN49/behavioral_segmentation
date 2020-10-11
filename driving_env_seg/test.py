import numpy as np


pic = np.zeros((100,100))

p1 = np.array([10,90])
p2 = np.array([80,40])

minp = np.array([min([p1[0],p2[0]]),min([p1[1],p2[1]])])
maxp = np.array([max([p1[0],p2[0]]),max([p1[1],p2[1]])])

vec1 = p1 - p2

vec1 = vec1/vec1[0]

print(vec1)
print(minp)

y = minp[1]
prey = int(round(y))

for x in range(pic.shape[0]):
    if maxp[0] >= x >= minp[0]:
        """
        for i in range(abs(prey-int(round(y)))):
            pic[x][prey+i+1] = 1"""
        pic[x][int(round(y))] = 1
        #prey = int(round(y))
        y = y + vec1[1]

print(pic)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

imgplot = plt.imshow(pic)
plt.show()