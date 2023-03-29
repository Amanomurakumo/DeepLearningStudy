import sys,os
sys.path.append(os.pardir)
import numpy as np

import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


# def getdata():
#     (x_train,t_train),(x_test,t_test)=\
#         load_mnist(normalize=True,flatten=True,one_hot_label=False)
#     return x_test,t_test

# def init_network():
#     with open("ch03/sample_weight.pkl",'rb')as f:
#         network=pickle.load(f)
#     return network

# def predict(network,x):
#     W1,W2,W3=network['W1'],network['W2'],network['W3']
#     b1,b2,b3=network['b1'],network['b2'],network['b3']
#     a1=np.dot(x,W1)+b1
#     z1=sigmoid(a1)
#     a2=np.dot(z1,W2)+b2
#     z2=sigmoid(a2)
#     a3=np.dot(z2,W3)+b3
#     y=softmax(a3)
#     return y

def numerical_gradient(f,x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
       
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 値を元に戻す
        
    return grad

def function_1(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

input=np.array([3.0,4.0])
grad=numerical_gradient(function_1,input)

    
# x,t=getdata()
# network=init_network()

# batch_size=100
# accuracy_cnt=0

# # for i in range(len(x)):
# #     y=predict(network,x[i])
# #     p=np.argmax(y)#ｙ配列内で、確率が一番大きい値のラベルをpに代入する
# #     if p==t[i]:#答えのラベルと比較して、正解していたら+1する
# #         accuracy_cnt+=1
# for i in range(0,len(x),batch_size):
#     x_batch=x[i:i+batch_size]
#     y_batch=predict(network,x_batch)
#     p=np.argmax(y_batch,axis=1)#ｙ配列内で、確率が一番大きい値のラベルをpに代入する
#     accuracy_cnt+=np.sum(p==t[i:i+batch_size])

# print("Accuracy:"+str(float(accuracy_cnt)/len(x)))       