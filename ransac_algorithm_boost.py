# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 19:27:13 2021

@author: fatcat
"""

import  numpy as np
import math as m
import random as r
import matplotlib.pyplot as plt

class parm():
      
    def __init__(self):
        
        self.first_cord = 0  # 第一組座標的index
        self.inlier = 0 
        self.outlier = 0
        self.itr = 1
        self.bestInNum = 0
        self.bestOutNum = 0
        self.best_A = 0 # ax+by+c = 0
        self.best_C = 0 # ax+by+c = 0
        self.outlier_distance = 20 # 與fitting直線距離超過20就視為outlier

# 正確率為 Predict 的情況 ， 資料點 M 個需要做 K 次 Ransac 才能到達該正確率 ，且假設每次抽取的資料點數 = t 。 註. M >> t
# inlier 機率 = p ，outlier 機率 = 1 - p  
# 抽取 1 次，t個點之中每一點都是inlier之機率 = p^t ， 因此有資料不為 inlier 的機率為 (1 - p^t)
# 抽取 K 次 = >  有資料不為 inlier 的機率為 (1 - p^t)^K
# 抽取 K 次 = > 每筆資料都是inlier的機率 Predict = 1 - (1 - p^t)^K
# Predict = 1 - (1 - p^t)^K ， 1 - P = (1 - p^t)^K
# 兩邊都取log => log(1 - Predict) = K * log(1 - p^t) ， K = log(1 - Predict) / log(1 - p^t)
#  K = log(1 - Predict) / log(1 - p^t) 
# 由於p通常不會提前知道 = > 需要估計

def setting():
    
    # 希望最後正確率
    Predict = float(input("Probability after RANSAC:")) 
    if Predict <= 0 or Predict >= 1:
        raise ValueError('Probability is between [0,1]')
        exit()
    
    # 估算整體資料內有9成是inlier => 抽到inlier機率 = 0.9
    p = float(input("Probability of inlier:")) 
    if p <= 0 or p >= 1:
        raise ValueError('Probability is between [0,1]')
        exit()
        
    t = 2  # 每次隨機抽取2個點
        
    M = float(input("Number of points:")) 
    M = int(M)
    if M <= 0 or M<= t:
        raise ValueError('Points < pick points')
        exit()
    if M % t > 0 :
        raise ValueError('M must be 2N')
        exit()
        
    K = m.log10(1 - Predict) / m.log10(1 - m.pow(p, t))
    K = round(K) # 次數必須是整數
        
    return K , M 
        
def Sampling(Itr_time , Total_points , cord_X , cord_Y ):
    
    for _ in range(Total_points):
        num1 = r.randint(0,2000)
        cord_X.append(num1)
        num2 = r.randint(0,2000)
        cord_Y.append(num2)
        
    pick = r.sample(range(150), Itr_time*2) # 要跑6次 一次需要2個點 每個數字都可以給x&y // ex pick[1] = 20 => X[20] Y[20] 是第一個點
    
    return pick , cord_X , cord_Y
    

def plot(X , Y):
    Coordinate = np.vstack((X,Y))
    plt.style.use("ggplot")
    plt.plot(Coordinate[0],Coordinate[1],'m.')
    plt.xlabel("X")
    plt.ylabel("Y")
    
def Ransac(Itr_time , X , Y , outlier_d, first_cord , inlier , outlier , itr , bestInNum , bestOutNum , best_A , best_C , pick , Total_point):
    
    while(Itr_time > 0):
        
        # select 2 point to form a line
        numx_1 , numy_1 = pick[first_cord] , pick[first_cord]
        numx_2 , numy_2 = pick[first_cord + 1] , pick[first_cord + 1]
        
        # 第一個點
        pick_X1 = X[numx_1]
        picK_Y1 = Y[numy_1]
        # 第二個點
        pick_X2 = X[numx_2]
        pick_Y2 = Y[numy_2]
        
        print("Pick Coordinate")
        print("X // Y")
        print(pick_X1,' ', picK_Y1)
        print(pick_X2,' ', pick_Y2)
        print("-----------")
        
        
        # 求直線  ax+by+c=0 slope = -a/b => b = 1 / a = -slope
        # 兩點斜率 
        slope = (pick_Y2 - picK_Y1)/(pick_X2 - pick_X1)
        a = -slope
        c = -(a*pick_X1 + 1*picK_Y1)
        
        for i in range(Total_point):
            top = abs( a*X[i] + 1*Y[i] + c)
            buttom = m.sqrt(a**2 + 1**2)
            distance = top/buttom
            if distance < outlier_d:
                inlier = inlier + 1
            else:
                outlier = outlier + 1
                
        print("number:",itr)
        print("Out:", outlier)
        print("In:", inlier)
        print("---------")
        
        # 紀錄目前最佳的直線
        if inlier > bestInNum:
            bestInNum , bestOutNum , best_A , best_C = inlier , outlier , a , c
            
        inlier = 0
        outlier = 0
        itr = itr + 1
        first_cord = first_cord + 2
        Itr_time = Itr_time - 1
    
    return bestInNum , bestOutNum , best_A , best_C

print("----Evaluate the Probability of Correctness----")
Itr_time , Total_point = setting()
r.seed(100)
X = []
Y = []
print("-----Sampling-----")
pick_list , X , Y = Sampling(Itr_time , Total_point , X , Y )
plot(X , Y)
print("-----Computing----")

# set P to class parm        
P = parm()
bestInNum , bestOutNum , best_A , best_C = Ransac(Itr_time , X , Y , P.outlier_distance , P.first_cord ,  P.inlier , P.outlier , P.itr , P.bestInNum , P.bestOutNum , P.best_A , P.best_C , pick_list , Total_point)
print("Best Formula : (",best_A,") X + Y + (",best_C,") = 0 ")
print("Best Inlier = ", bestInNum)
print("Best Outlier = ", bestOutNum)



































