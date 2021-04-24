# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:53:27 2021

@author: fatcat
"""

import  numpy as np
import math as m
import random as r
import matplotlib.pyplot as plt

print("----Evaluate the Probability of Correctness----")

# 正確率為 Predict 的情況 ， 資料點 M 個需要做 K 次 Ransac 才能到達該正確率 ，且假設每次抽取的資料點數 = t 。 註. M >> t
# inlier 機率 = p ，outlier 機率 = 1 - p  
# 抽取 1 次，t個點之中每一點都是inlier之機率 = p^t ， 因此有資料不為 inlier 的機率為 (1 - p^t)
# 抽取 K 次 = >  有資料不為 inlier 的機率為 (1 - p^t)^K
# 抽取 K 次 = > 每筆資料都是inlier的機率 Predict = 1 - (1 - p^t)^K
# Predict = 1 - (1 - p^t)^K ， 1 - P = (1 - p^t)^K
# 兩邊都取log => log(1 - Predict) = K * log(1 - p^t) ， K = log(1 - Predict) / log(1 - p^t)
#  K = log(1 - Predict) / log(1 - p^t) 
# 由於p通常不會提前知道 = > 需要估計


Predict = 0.99 # 希望最後正確率 99%

p = 0.75 # 估算整體資料內有9成是inlier => 抽到inlier機率 = 0.9

t = 2  # 每次隨機抽取2個點

# t 必須是資料總數的因素 t* n = M


K = m.log10(1 - Predict) / m.log10(1 - m.pow(p, t))

K = round(K) # 次數必須是整數

print(K)  

# 選定 outlier 大小
outlier_distance = 20 # 與fitting直線距離超過20就視為outlier

# 製造隨機座標點
M = 150 # 200個座標點

if M % t > 0 :
    raise ValueError('Failed in spliting group!!')
    exit()

r.seed(100)
X = []
Y = []
for _ in range(M):
    num1 = r.randint(0,200)
    X.append(num1)
    num2 = r.randint(0,200)
    Y.append(num2)

Coordinate = np.vstack((X,Y))
# print(Coordinate[1])
# print(Coordinate[1][0])
plt.style.use("ggplot")
plt.plot(Coordinate[0],Coordinate[1],'m.')
plt.xlabel("X")
plt.ylabel("Y")

print("-----Computing-----")

# 挑選隨機的座標
pick = r.sample(range(150), K*2) # 要跑6次 一次需要2個點 每個數字都可以給x&y // ex pick[1] = 20 => X[20] Y[20] 是第一個點
first = 0
second = 1
# print(pick)

# 紀錄參數
inlier = 0
outlier = 0
itr = 1
bestInNum = 0
bestOutNum = 0
best_A = 0
best_C = 0

print('-------')

while(K > 0):
    # select 2 point to form a line
    numx_1 = pick[first]
    numy_1 = pick[first]
    numx_2 = pick[second]
    numy_2 = pick[second]
    
    # 第一個點
    pick_X1 = X[numx_1]
    picK_Y1 = Y[numy_1]
    # 第二個點
    pick_X2 = X[numx_2]
    pick_Y2 = Y[numy_2]
    
    # print(pick_X1,' ', picK_Y1)
    # print(pick_X2,' ', pick_Y2)
    # print("-----------")
    
    # 求直線  ax+by+c=0 slope = -a/b => b = 1 / a = -slope
    # 兩點斜率 
    slope = (pick_Y2 - picK_Y1)/(pick_X2 - pick_X1)
    a = -slope
    c = -(a*pick_X1 + 1*picK_Y1)
    #得方程式 ax+by+c = 0
    # 點到距離公式 : 絕對值(點帶入方程式) / sqrt(a**2+b**2)
    for i in range(M):
        top = abs( a*X[i] + 1*Y[i] + c)
        buttom = m.sqrt(a**2 + 1**2)
        distance = top/buttom
        if distance < outlier_distance:
            inlier = inlier + 1
        else:
            outlier = outlier + 1
    
    print("number:",itr)
    print("Out:", outlier)
    print("In:", inlier)
    print("---------")
    
    # 紀錄目前最佳的直線
    if inlier > bestInNum:
        bestInNum = inlier
        bestOutNum = outlier
        best_A = a
        best_C = c
    
    inlier = 0
    outlier = 0
    itr = itr + 1
    first = first + 2
    second = second + 2
    K = K - 1
    
print("Best Formula : (",best_A,") X + Y + (",best_C,") = 0 ")
print("Best Inlier = ", bestInNum)
print("Best Outlier = ", bestOutNum)























