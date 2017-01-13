import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5, 0.1);
y = x**2 
plt.plot(x, y)
x=np.array([1,2,3])
y=x**2
plt.scatter(x,y)
plt.show()
