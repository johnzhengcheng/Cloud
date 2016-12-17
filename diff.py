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

from sympy import *
a,b,m,n,x, y,z,t,k= symbols('a b m n x y z t k')
init_printing(use_unicode=True)


print("Please input the function")



maths_expression=input()

diff_expression=simplify(diff(maths_expression,x))

print("The difference of "+maths_expression+" is ")

print(diff_expression)
