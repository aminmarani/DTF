import os
from concurrent.futures import ThreadPoolExecutor
import ast


executor = ThreadPoolExecutor(30)
def func(name):
    os.system('ssh -i "/home/mohamed/Downloads/mykey.pem" ' + name + ' sudo rm -r /home/ec2-user/RHT')
    os.system("scp -i /home/mohamed/Downloads/mykey.pem -r /home/mohamed/Desktop/OperatingSystems/RHT " + name + ":/home/ec2-user/RHT")


def func2(name,send_chunks):
    cluster = ast.literal_eval(name)
    workers=cluster["worker"]
    master=cluster["master"]
    for i,worker in enumerate(workers):
        if send_chunks:
            os.system('ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' sudo rm -r /home/ec2-user/DistributedTF')
            os.system('ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' sudo mkdir -m 777 /home/ec2-user/DistributedTF')
            os.system('ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' sudo mkdir -m 777 /home/ec2-user/DistributedTF/chunks')
            os.system(
                "scp -i /home/mohamed/Downloads/mykey.pem -r /home/mohamed/Desktop/OperatingSystems/DistributedTF/chunks/chunk"+str(i)+".obj " + worker + ":/home/ec2-user/DistributedTF/chunks/chunk"+str(i)+".obj")
            os.system(
                "scp -i /home/mohamed/Downloads/mykey.pem -r /home/mohamed/Desktop/OperatingSystems/DistributedTF/chunks/chunk_labels" + str(i) + ".obj " + worker + ":/home/ec2-user/DistributedTF/chunks")

        os.system('ssh -i "/home/mohamed/Downloads/mykey.pem" ' + worker + ' sudo rm -r /home/ec2-user/DistributedTF/code')
        os.system("scp -i /home/mohamed/Downloads/mykey.pem -r /home/mohamed/Desktop/OperatingSystems/DistributedTF/code " + worker + ":/home/ec2-user/DistributedTF/code")

    if send_chunks:
        os.system('ssh -i "/home/mohamed/Downloads/mykey.pem" ' + master[0] + ' sudo rm -r /home/ec2-user/DistributedTF')
        os.system(
            'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + master[0] + ' sudo mkdir -m 777 /home/ec2-user/DistributedTF')
        os.system(
            'ssh -i "/home/mohamed/Downloads/mykey.pem" ' + master[0] + ' sudo mkdir -m 777 /home/ec2-user/DistributedTF/chunks')
        os.system(
            "scp -i /home/mohamed/Downloads/mykey.pem -r /home/mohamed/Desktop/OperatingSystems/DistributedTF/chunks/chunk_master.obj " + master[0] + ":/home/ec2-user/DistributedTF/chunks/chunk_master.obj")
        os.system(
            "scp -i /home/mohamed/Downloads/mykey.pem -r /home/mohamed/Desktop/OperatingSystems/DistributedTF/chunks/chunk_master_labels.obj " + master[0] + ":/home/ec2-user/DistributedTF/chunks/chunk_master_labels.obj")

        os.system(
            "scp -i /home/mohamed/Downloads/mykey.pem -r /home/mohamed/Desktop/OperatingSystems/DistributedTF/chunks/chunk_master.obj " +
            master[0] + ":/home/ec2-user/DistributedTF/chunks/chunk_validation.obj")
        os.system(
            "scp -i /home/mohamed/Downloads/mykey.pem -r /home/mohamed/Desktop/OperatingSystems/DistributedTF/chunks/chunk_master_labels.obj " +
            master[0] + ":/home/ec2-user/DistributedTF/chunks/chunk_validation_labels.obj")
    os.system('ssh -i "/home/mohamed/Downloads/mykey.pem" ' + master[0] + ' sudo rm -r /home/ec2-user/DistributedTF/code')
    os.system(
        "scp -i /home/mohamed/Downloads/mykey.pem -r /home/mohamed/Desktop/OperatingSystems/DistributedTF/code " + master[0] + ":/home/ec2-user/DistributedTF/code")


f = open("distributed_servers_names", "r")
names=[]
for x in f:
    if x[-1]=="\n":
        names.append(x[:-1])
    else:
        names.append(x)

send_chunks=False
for i in range(len(names)):
    #executor.submit(func,names[i])
    func2(names[i],send_chunks)




