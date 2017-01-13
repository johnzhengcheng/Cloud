import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 5, 0.1);
y = x**2+2*x+1 
plt.plot(x, y,linestyle='-.',color='red')
x=np.array([1,2,3,4,5])
y=x**2+2*x+1
plt.scatter(x,y,linewidths=5)
plt.annotate('Point 1 ',xy=(1,4),arrowprops=dict(facecolor='blue', shrink=0.05))  
plt.annotate('Point 5 ',xy=(5,5**2+2*5+1),xytext=(4.96,31.4),arrowprops=dict(facecolor='blue', shrink=0.05))

plt.show()
