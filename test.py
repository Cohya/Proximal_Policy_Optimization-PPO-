from threading import Thread
from multiprocessing import Process
import multiprocessing

def func2(x,y):
    print(str(x))
    y.append(x+1)
    
    return x+ 1

def func(x):
    # print(str(x))
    # y.append(x+1)
    return x+ 1

x = [1,2,3,4]
y = []

for i in range(len(x)):
    Thread(target = func2, args = ([x[i], y])).start()
    

