import os
from concurrent.futures import ThreadPoolExecutor
import ast


executor = ThreadPoolExecutor(30)

def func2(name):
    cluster = ast.literal_eval(name)
    workers=cluster["worker"]
    master=cluster["master"]
    for i,worker in enumerate(workers):
        os.system('ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py')
        os.system(
            'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' sudo python get-pip.py')
        os.system(
            'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' sudo pip install -U virtualenv')
        os.system(
            'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' virtualenv --system-site-packages -p python2.7 /home/ec2-user/venv')
        # os.system(
        #     'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' source ~/venv/bin/activate')
        # os.system(
        #     'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' sudo pip install --upgrade tensorflow')
        os.system(
            'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' \'source /home/ec2-user/venv/bin/activate && sudo pip install --upgrade tensorflow' + '\'')

    os.system(
        'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + master[0] + ' curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py')
    os.system(
        'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + master[0] + ' sudo python get-pip.py')
    os.system(
        'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + master[0] + ' sudo pip install -U virtualenv')
    os.system(
        'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + master[0] + ' virtualenv --system-site-packages -p python2.7 /home/ec2-user/venv')
    os.system(
        'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + master[0] + ' \'source /home/ec2-user/venv/bin/activate && sudo pip install --upgrade tensorflow'+'\'')
    # os.system(
    #     'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + master[0] + ' sudo pip install --upgrade tensorflow')


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