import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# data preprocessing
def data_preprocessing():
    df=pd.read_csv('./data/train.csv',usecols=range(3,27),encoding='utf-8')

    df=df.replace(['NR'],[0.0])

    arr=np.array(df).astype(float)
    x,y=[],[]
    for i in range(0,4320,18):
        for j in range(24-9):
            mat=arr[1:1+18,j:j+9]
            label=arr[i+9,j+9]
            x.append(mat)
            y.append(label)
    x=np.array(x)
    y=np.array(y)

    return x,y,arr

# train

def train(x,y,epoch):
    bias=0
    weight=np.ones(9)

    learning_rate=0.001

    reg_rate=0.001

    bg_sum=0
    wg_sum=np.zeros(9)
    loss_sum=[]
    count=[]
    for i in range(epoch):
        b_grad = 0.0
        w_grad = 0.0
        for j in range(3200):
            b_grad = b_grad -(y[j]-weight.dot(x[j,9,:])-bias)*(1)
            for n in range(9):
                w_grad=w_grad-(y[j]-weight.dot(x[j,9,:])-bias)*(x[j,9,n])

        b_grad/=3200.0
        w_grad/=3200.0

        for m in range(9):
            w_grad+=reg_rate*weight[m]

        bg_sum=bg_sum+b_grad**2
        wg_sum=wg_sum+w_grad**2

        # update paramters
        bias-=learning_rate/bg_sum**0.5 * b_grad
        weight-=learning_rate/wg_sum**0.5*weight

        if i % 500==0:
            loss=0
            for j in range(3200):
                loss+=(y[j]-weight.dot(x[j,9,:])-bias)**2
            print('%d epoches,loss=%s' % (i,loss/3200))
            loss_sum.append(loss/3200)
            count.append(i)
    return weight,bias,count,loss_sum

def validate(x,y,w,b):
    loss=0
    for i in range(400):
        loss+=(y[i]-w.dot(x[i,9,:])-b)**2
    return loss/400

x, y, _ = data_preprocessing()
x_train, y_train = x[0:3200], y[0:3200]
x_val, y_val = x[3200:3600], y[3200:3600]
epoch = 10000*10

w, b,count,loss_sum= train(x_train, y_train, epoch)

loss = validate(x_val, y_val, w, b)
print('The loss on val data is:', loss)


plt.plot(count,loss_sum)
plt.show()