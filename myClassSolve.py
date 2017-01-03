# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a mathematics script file.
Which is used for solving an equeation.
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

import myClass

from sympy import *
a,b,m,n,x, y,z,t,k= symbols('a b m n x y z t k')
init_printing(use_unicode=True)

class mySolve(myClass.myData):    
         
         

    def myOutput(self,indicate):        
        if indicate=='D':
            print("The answers are below:")
            for i in self.maths_output:
                print(round(i.evalf(),2))
        else:
            super(mySolve,self).myOutput()

    def myOutput(self):
        print("The answers are below:")
        for i in self.maths_output:
            print(round(i.evalf(),2))
                    
'''
    
    def mySolve(self):
        self.position=self.maths_exp.split("=")
        self.tmp_str=self.position[0]+"-("+self.position[1]+")"
        self.maths_output=solve(self.tmp_str,x)
        
    
    
        
        
  
  

      
      
  
	
    
  
  

  
   