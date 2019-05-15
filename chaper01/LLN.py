import random
import matplotlib.pyplot as plt

def random_plot(minExp,maxExp):
    '''
    :param minExp: 
    :param maxExp: 
    :return: 
    抛硬币次数为2的minExp次方到2的maxExp次方,即一共做了2^maxExp-2^minExp批次实验,每批次重复抛硬币2^n次
    '''
    ratios=[]
    xAxis=[]
    for exp in range(minExp,maxExp+1):
        xAxis.append(2**exp)
    for batch in xAxis:
        countHeads=0
        for n in range(batch):
            if random.random()<0.5:
                countHeads+=1
        countTails=batch-countHeads
        ratios.append(countHeads/countTails)
    plt.title('Heads/Tails Ratios')
    plt.xlabel('Number of Flips')
    plt.ylabel('Heads/Tails')

    plt.plot(xAxis,ratios)
    plt.hlines(1,0,xAxis[-1],linestyles='dashed',colors='r')
    plt.show()
random_plot(4,16)