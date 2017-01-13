import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5, 0.1);
y = x**2+2*x+1 
plt.plot(x, y,linestyle='-.',color='red')
x=np.array([1,2,3])
y=x**2+2*x+1
plt.scatter(x,y,linewidths=5)
plt.show()
