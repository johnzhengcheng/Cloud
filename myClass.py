# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a mathematics script file.
Which is used for deriving the difference of each function.
"""

'''
Created on Dec 17,2016

@author: Zheng Cheng
'''
'''
使用方法
t=myClass.myData('t')
t.myInput()
t.myOutput()
'''

from sympy import *
a,b,m,n,x, y,z,t,k= symbols('a b m n x y z t k')
init_printing(use_unicode=True)


class myData:    
    def __init__(self,tag):
        self.title=tag
        self.maths_exp=""
        self.maths_deriv=""
        self.maths_integrate=""
        self.maths_output=""
        
    def myInput(self):
        print("Please input your maths expression:")
        self.maths_exp=input()
        
    def myOutput(self):
        pprint(self.maths_output,use_unicode=False)
        
    def myValue(self,value):
        self.maths_output=value
        
    
            
        
        
  
  

      
      
  
	
    
  
  

  
   