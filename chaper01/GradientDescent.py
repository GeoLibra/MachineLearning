import numpy as np
import matplotlib.pyplot as plt
x_data=[338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data=[640.,633.,619.,393.,428.,27.,193.,66.,226.,1591.]

bias=np.arange(-200,-100,1) # bias
weight=np.arange(-5,5,0.1)    # weight
Z=np.zeros((len(bias),len(weight)))
X,Y=np.meshgrid(bias,weight)
for i in range(len(bias)):
    for j in range(len(weight)):
        b=bias[i]
        w=weight[j]
        Z[j][i]=0
        for n in range(len(x_data)):
            Z[j][i]=Z[j][i]+(y_data[n]-b-w*x_data[n])**2
        Z[j][i]=Z[j][i]/len(x_data)
#ydata=b+w*x_data
b=-120.0
w=-4.0
lr=1 # learning rate
iteration=100000

lr_b=0
lr_w=0

# store initial values for plotting
b_history=[b]
w_history=[w]

#Iterations
for i in range(iteration):
    b_grad=0.0
    w_grad=0.0
    for n in range(len(x_data)):
        b_grad=b_grad-2.0*(y_data[n]-b-w*x_data[n])*1.0
        w_grad=w_grad-2.0*(y_data[n]-b-w*x_data[n])*x_data[n]
    # update parameters
    lr_b=lr_b+b_grad**2
    lr_w=lr_w+w_grad**2

    b=b-lr/np.sqrt(lr_b)*b_grad
    w=w-lr/np.sqrt(lr_w)*w_grad

    b_history.append(b)
    w_history.append(w)

plt.contourf(bias,weight,Z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms=6,marker=6,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()
