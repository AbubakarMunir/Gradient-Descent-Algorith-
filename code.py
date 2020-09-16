import xlrd
import xlwt
import random
import sys
import pandas as pd
import math
import numpy
import matplotlib.pyplot as p

#this function reads the randomized excel file into a dataset container and returns it for later use
def ReadFile():
    excel_file = "iris dataset.xlsx"
    dataset = pd.read_excel(excel_file)
    for i in range(0, 150):
        label = dataset.iloc[i, 4]
        if label == "I. versicolor":
            dataset.iloc[i, 4] = 1
        if label == "I. virginica":
            dataset.iloc[i, 4] = 2
        if label == "I. setosa":
            dataset.iloc[i, 4] = 3
    return dataset

#cost function
def calcCost(dataset,w):
    cost=0
    for i in range(0, 100):
        ypred=w[0] + w[1]*dataset.iloc[i, 0] + w[2]*dataset.iloc[i, 1] + w[3]*dataset.iloc[i, 2] + w[4]*dataset.iloc[i, 3]
        y= dataset.iloc[i, 4]
        costsquare= (y - ypred) * (y - ypred)
        cost=cost + costsquare
    cost=cost/2
    return cost

#this function returns the the summation of (y-ypred)*Ik ie derivative of cost wrt to each weight separarately
def calcCostSum(dataset,w):

    costsum=[0,0,0,0,0]
    for i in range(0, 100):
        ypred = w[0] + w[1] *dataset.iloc[i, 0] + w[2] * dataset.iloc[i, 1] + w[3] * dataset.iloc[i, 2] + w[4] * dataset.iloc[i, 3]
        y = dataset.iloc[i, 4]
        diff=y - ypred
        #derivative of each weight
        costsum[0]=costsum[0] +diff
        costsum[1]=costsum[1]+ (diff*dataset.iloc[i, 0])
        costsum[2] = costsum[2] + (diff * dataset.iloc[i, 1])
        costsum[3] = costsum[3]+ (diff * dataset.iloc[i, 2])
        costsum[4] = costsum[4] + (diff * dataset.iloc[i, 3])

    return costsum


#GRADIENT DESCENT FUNCTION
def GradientDescent(dataset,w):

    trRate=0.0001
    iter=0
    cost=100          #an arbitrary val just to enter while loop
    costarr=[]
    while(iter<=1000 and cost!=0 ):
        costsum = calcCostSum(dataset, w)
        #c0 to c5 are values of derivatives of c wrt to every weight
        c0=costsum[0]
        c1 = costsum[1]
        c2 = costsum[2]
        c3 = costsum[3]
        c4 = costsum[4]

        w[0] = w[0] - (trRate * (-1) * (c0) ) # input is 1 for bias so no input
        w[1] = w[1] - trRate * (-1) * (c1)
        w[2] = w[2] - trRate * (-1) * (c2)
        w[3] = w[3] - trRate * (-1) * (c3)
        w[4] = w[4] - trRate * (-1) * (c4)

        iter=iter+1
        if(iter %100==0):
            cost = calcCost(dataset, w)
            costarr.append(cost)                #the array consisting of cost values which will be used later
            #print(cost)
    return costarr

#to return the number of mismatches while testing
def Testmismatches(dataset,w):
    mismatches=0
    for i in range(100, 150):
        d=dataset.iloc[i,4]
        #print("y=",d)
        ypred = w[0] + w[1] * dataset.iloc[i, 0] + w[2] * dataset.iloc[i, 1] + w[3] * dataset.iloc[i, 2] + w[4] * dataset.iloc[i, 3]
        if(ypred<=1.5):
            ypred=1
        if (ypred>1.5 and ypred<=2.5):
            ypred = 2
        if (ypred>2.5):
            ypred = 2
        if(d!=ypred):
            mismatches=mismatches+1
        #print("ypred=",ypred)
    return mismatches







w= [2,1,0,1,0]
a=ReadFile()
s=GradientDescent(a,w)

#to plot our data
y=[100,200,300,400,500,600,700,800,900,1000]
p.plot(y,s)
p.show()

#testing on remaining samples
count=Testmismatches(a,w)
print(count)




