import os
from concurrent.futures import ThreadPoolExecutor
import ast


executor = ThreadPoolExecutor(30)

def func2(name):
    cluster = ast.literal_eval(name)
    workers=cluster["worker"]
    master=cluster["master"]
    for i,worker in enumerate(workers):
        os.system('ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' sudo pip install -U scikit-learn')


    os.system(
        'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + master[0] + ' sudo pip install -U scikit-learn')



f = open("distributed_servers_names", "r")
names=[]
for x in f:
    if x[-1]=="\n":
        names.append(x[:-1])
    else:
        names.append(x)

for i in range(len(names)):
    #executor.submit(func,names[i])
    func2(names[i])