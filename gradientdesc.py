import numpy as np
import pandas as pd
m=0
c=0
lr=0.00001
epoch=10000
mydataset=pd.read_csv('aimarks2017.csv')
x=mydataset['mse'].values
y=mydataset['ese'].values
n=float(len(x))
for i in range(epoch):
    y_pred=x*m+c
    dm=(-2/n)*sum(x*(y-y_pred))
    de=(-2/n)*sum(y-y_pred)
    m=m-lr*dm
    c=c-lr*de
print("Using gradient descent algo:")
print(m,c)
a=pd.Series(x)
b=pd.Series(y)
r=a.cov(b)/(a.std()*b.std())
beta1=(r*b.std())/a.std()
beta0=b.mean()-beta1*a.mean()
print("Using OSE Method:")
print(beta1,beta0)
