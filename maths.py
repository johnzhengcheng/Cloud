# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a mathematics script file.
"""

'''
Created on Aug 24, 2015

@author: echezhe
'''

from sympy import *
a,b,m,n,x, y,z,t,k= symbols('a b m n x y z t k')
init_printing(use_unicode=True)



sqrt(8)

print(integrate(sin(x**2)))

print(diff(x**2))


###计算化简公式


print(simplify((x+y)*(x**2-x*y+y**2)))

print(integrate(exp(-x**2 - y**2), (x, -oo, oo), (y, -oo, oo)))

###计算不定积分
print(integrate(x))


##计算极限
print(limit((cos(x)-1)/x,x,0))

##解方程

print(solve(x**2-1,x))

#泰勒级数

print(sin(x).series(x,0,10))


##微分方程

from sympy import Function, Symbol, dsolve
f = Function('f')
x = Symbol('x')
dsolve(f(x).diff(x, x) + f(x), f(x))
#f(x) = C₁⋅cos(x) + C₂⋅sin(x)

#采样还原

#去正弦函数pi/2和pi*3/2

plot(sin(2*(x-pi/2))/(2*(x-pi/2))-sin(2*(x-pi*3/2))/(2*(x-pi*3/2)))


######计算方程组

from scipy.optimize import fsolve
from math import sin,cos

def f(x):
    x0 = float(x[0])
    x1 = float(x[1])
    return [
    3*x0+2*x1-7,x0+x1-9]


result=fsolve(f,[1,1])

print(result)

###如何绘制方波，将若干正弦波叠加，生成矩形波
plot(0.5+0.637*cos(x)-0.212*cos(3*x)+0.127*cos(5*x))


###计算卷积
a1=[1,1]
a2=[1,2,5]
import numpy as np
np.convolve(a1,a2)
array([1, 3, 7, 5])


###########三角函数
expand_trig(sin(a+b))
###￼￼￼sin(a)cos(b)+sin(b)cos(a)

###阶乘###
factorial(n)

###C 4,2###
binomial(4,2)

####展示整个公式    
Integral(cos(x)*exp(x), x)
Eq(a,a.doit())

###多项式展开
expand()

###多项式合并
factor()

###解方程组
solve([x - y + 2, x**2-y-1], [x, y])


###获得矩阵的秩,逆矩阵
#单位阵列
i=numpy.eye(4)
##普通阵列
a=Matrix([[2,2],[3,4]])

###矩阵的秩
numpy.linalg.matrix_rank(i)

numpy.￼linalg.inv(a)


###############
注意：在这些输出值中，第一个值是对应的直流分量的振幅（其实就是周期为无穷的
可能性），那么第2个值对应第1个采样点，第3个对应第2个。。
第n个对应第n-1个采样点。而且这个输出是对称的，
也就是大家直接关注前N/2个才样点就可以了。
那么第n个点的频率是多少呢，
它的计算公式是Fn=(n-1)*Fs/N，其中Fs是采样频率。由此就可以计算出n点对应的周期了，
它是频率的倒数，即Tn=N/((n-1)*Fs)。下面给出两个例子：
例一：>>A=[1,2,1,2,1,2];
          >>fft(A)
           ans= 9 0 0 -3 0 0
         这里输出的意思是，序列A有很大的可能没有周期
         （第一个点的频率为0，它对应的数字是9），
         还有一个可能的周期是-3对应的周期，
         这个周期的计算方法是:-3对应于n=4,默认Fs=1,
         这里T=6/(3*1)=2,即周期为2。看明白了吗？
￼

========FFT图形
import numpy as np
import matplotlib.pyplot as pl
sampling_rate = 8000
fft_size = 512
t = np.arange(0, 1.0, 1.0/sampling_rate)
x = np.sin(2*np.pi*156.25*t)  + 2*np.sin(2*np.pi*234.375*t)
#x=6*np.sin(2*np.pi*100*t)+3*np.sin(2*np.pi*200*t)
xs = x[:fft_size]
xf = np.fft.rfft(xs)/fft_size
freqs = np.linspace(0, sampling_rate/2, fft_size/2+1)
xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
#xfp=np.abs(xf)*2/fft_size
#pl.figure(figsize=(8,4))
pl.subplot(211)
pl.plot(t[:fft_size], xs)
pl.xlabel(u"Time(Sec)")
pl.title(u"156.25Hz and 234.375Hz")
pl.subplot(212)
pl.plot(freqs, xfp)
pl.xlabel(u"Frequency(Hz)")
pl.subplots_adjust(hspace=0.4)
pl.show()
===========================================

##KNN算法
from sklearn import neighbors
import numpy as np
knn=neighbors.KNeighborsClassifier()
#data=np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
data=np.array([[80,75,66],[85,80,33],[70,70,22],[90,91,77],[99,98,44],[95,96,55]])
labels=np.array([[1,1],[1,2],[1,0],[2,2],[2,1],[2,0]])
knn.fit(data,labels)
knn.predict([18,90,33])

##数据固定化
from sklearn.externals import joblib
joblib.dump(knn,'c:\\backup\\knn.pkl')

##提取模型
knn2=joblib.load('c:\\backup\\knn.pkl')
knn2.predict([18,80,33])

===========================================
numpy FFT


###假设基准频率是f,采样频率是4f
import numpy as np
A=np.array([0,1,0,-1,0,1,0,-1,])
B=np.fft.fft(A)

===B
array([ 0.+0.j,  0.+0.j,  0.-4.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+4.j,
        0.+0.j])
Fn=(n-1)*Fs/N
第2项最大：频率为2*4/8
相位为-pi/2
cos(x-pi/2)

np.fft.ifft(B)

===============================

散点画图的方法

import numpy as np
import matplotlib.pyplot as plt

n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)

plt.scatter(X,Y)
plt.show()

#####
其他画图

pl.subpl


#########################3
线性回归

from numpy import arange,array,ones,linalg
from pylab import plot,show


xi = arange(0,9)
xj = arange(11,20)

A = array([ xi, xj, ones(9)])

# linearly generated sequence
y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]

print(y)
w = linalg.lstsq(A.T,y)[0] # obtaining the parameters

print(A.T)



print(w.shape)
print(w[0],w[1],w[2])



# plotting the line

line = w[0]*xi+w[1]*xj+w[2] # regression line

print(line)
#plot(xi,line,'r-',xi,y,'o')

#plot(xi,xj,line,'r-')
#show()

=============================正态分布

from scipy import stats
from scipy.stats import poisson
from scipy.stats import expon
from scipy.stats import binom


#####t-tset   
stats.t.ppf(q=0.025,df=34)  ###df是自由度

###正态分布

print(stats.norm.cdf(5,1,4)-stats.norm.cdf(-3,1,4))

print((1-(stats.norm.cdf(6)-stats.norm.cdf(-6))))

###z值为 样本均值x1,
###

求z值，
stats.norm.ppf(0.975)
1.96

###泊松分布，0是k, 3是lamda
r=poisson.pmf(0,3)**2
print(r)


###指数分布，0.25是x, lamda是3
r=expon.cdf(0.25,scale=1/3)
print(r)

r=expon.cdf(0.5,scale=1/3)-expon.cdf(0.25,scale=1/3)
print(r)


==============多元函数积分
from scipy import integrate
from math import sin


func = lambda x,y : y*sin(x)

print(integrate.nquad(func, [[0,1], [0,1]]))

====================读取CSV文件
import csv
reader = csv.reader(open("c:\\users\\echezhe\\desktop\\test1.csv"))
for title in reader:
    print(title[0],title[1],int(title[0])+int(title[1]))
    print(len(title))
 
    
====================sklearn做线性回归
from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
clf.coef_
clf.intercept_

=====================Pands
#####df新创建一个列
import pandas as pd
df=pd.read_csv("c:\\backup\\tips.csv")

#########提取部分字段
df_input=df.iloc([0,1])

df1['newcolumnname']='value'

df1=df1.convert_objects(convert_numeric=True)



###########SVM
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)  

#####drop df column,drop df
del df1["sex"]
del df1
df1=df.ix[(df.radio == 'LTE') & (df.mcc!=460)]
In [2]: df.ix[df.AAA >= 5,'BBB'] = -1; df
Out[2]: 
   AAA  BBB  CCC
0    4   10  100
1    5   -1   50
2    6   -1  -30
3    7   -1  -50


###########John Zheng改写FFT
import numpy as np
import matplotlib.pyplot as pl
sampling_rate = 2000
#fft_size = 512
t = np.arange(0, 1.0, 1.0/sampling_rate)
#x=np.cos(2*np.pi*50*t)*np.cos(2*np.pi*30*t)*np.cos(2*np.pi*30*t)
x= 2+3*np.cos(2*np.pi*50*t-np.pi*30/180)+1.5*np.cos(2*np.pi*75*t+np.pi*90/180)
#x=np.zeros(256)
#x=x+5
#x=2*np.cos(2*np.pi*10*t)
'''
假设我们有一个信号，它含有a、2V的直流分量，b、频率为50Hz、相位为-30度、幅度为3V的交流信号，以及c、频率为75Hz、相位为90度、幅度为1.5V的交流信号。用数学表达式就是如下：
  S=2+3*cos(2*pi*50*t-pi*30/180)+1.5*cos(2*pi*75*t+pi*90/180)。
'''
xf = np.fft.fft(x)
xfmode=np.abs(np.fft.fft(x))*2/sampling_rate
xfmode[0]=xfmode[0]/2
pl.subplot(211)
pl.plot(x)
pl.subplot(212)
pl.axis([-3,200,0,5])
pl.plot(xfmode[:sampling_rate/2])
pl.show()
###调整显示横轴和纵轴
#求相位
from sympy import *
print(atan(np.imag(xf[50])/np.real(xf[50])))
print(atan(np.imag(xf[75])/np.real(xf[75])))

####建议画图
t=np.arange(-1,10,1)
m=t**3+t**2
pl.plot(t,m)

##########第一类线积分
from sympy import Curve, line_integrate
from sympy.abc import x,y,t
C=Curve([t,t],(t,0,1))  ###定义曲线
line_integrate(1,C,[x,y])



#####文件改名
import os,sys

dir="i:\\misc\\"

filenames=os.listdir(dir)

for a in filenames:
    #b=a.replace("[电影天堂www.dy2018.com]","")
    t=a.find("】")
    if(t>0):
        b=a[t+1:]
        os.rename(dir+os.sep+a,dir+os.sep+b)
    t=a.find("]")
    if(t>0):
        b=a[t+1:]
        os.rename(dir+os.sep+a,dir+os.sep+b)
        
###FM例子
plot(cos(integrate(x)))
plot(cos(integrate(t,(t,0,x))),(x,0,20))


#####mysql操作
import pymysql
conn = pymysql.connect(host='localhost', port=3306,user='root',passwd='hello123',db='App',charset='UTF8')
cur = conn.cursor()
cur.execute("select * from t_app")
for i in cur:
    print(i[1])
conn.close()

######list的操作,filter的操作
a=[3,4,5]
b=[index1**2 for index1 in a]


b=filter(lambda x:x>5 and x%7==0,range(20))
b=list(b)

c=[3,7,5]

###########拉格朗日插值法
from scipy.interpolate import lagrange
return lagrange(y.index, list(y))(n) #插值并返回插值结果
###去掉空值
 y = y[y.notnull()] #剔除空值


####矩阵乘法
np.dot

###chunksize的处理
df=pd.read_csv('train_ver2.csv',sep=',',chunksize=10**5)


In [38]: for chunk in df:
    print(chunk['age'])

import pandas as pd
df=pd.read_csv('train_ver2.csv',chunksize=5*10**5)
a,b,c=[],[],[]
for chunk in df:
    chunk1=chunk.convert_objects(convert_numeric=True)
    a.append(chunk1['age'].max())
    b.append(chunk1['age'].min())
    c.append(chunk1['age'].mean())

a.sort()
b.sort()
c.sort()
#Add some data
#Add some data



