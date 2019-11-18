import threading
import socket
import sys

from concurrent.futures import ThreadPoolExecutor
import logging
import time
import itertools
import os
import ast
import json
import os
import pickle
import sys
import distributed_training

counter = itertools.count()
period=3


import time, threading

def func(worker,id):
    os.system('python start_training_for_workers.py '+worker+' '+str(id))

def func2(worker,id,worker_name):
    #os.system('python start_training_for_workers.py '+worker+' '+str(id))

    #os.system('ssh -i "/home/ec2-user/DistributedTF/code/mykey.pem" ' + worker_name + ' python start_training_for_workers.py '+worker+' '+str(id))
    os.system('ssh -i "/home/ec2-user/DistributedTF/code/mykey.pem" ' + worker_name + ' \'cd /home/ec2-user/DistributedTF/code && python start_training_for_workers.py '+worker+' '+str(id)+'\'')

def clear_port(port,worker_name):
    #os.system('ssh -i "/home/ec2-user/DistributedTF/code/mykey.pem" ' + worker_name + ' fuser -n tcp -k '+port)
    os.system('ssh -i "/home/ec2-user/DistributedTF/code/mykey.pem" ' + worker_name + ' \'sudo fuser -k ' + port + '/tcp' + '\'')



def handle(connection, address):


    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("process-%r" % (address,))

    try:
        logger.debug("Connected %r at %r", connection, address)


        while True:
            data = connection.recv(1024)
            if data.decode() == "":
                logger.debug("Socket closed remotely")
                break
            logger.debug("Received data %r", data)

            splitted=data.decode().split(" ")
            if splitted[0]=="TRAIN":
                file_name = "distributed_servers"
                file = open(file_name, "r")
                cluster = file.read()
                cluster = ast.literal_eval(cluster)
                workers=cluster["worker"]

                file_name2 = "distributed_servers_names"
                file2 = open(file_name2, "r")
                cluster2 = file2.read()
                cluster2 = ast.literal_eval(cluster2)
                worker_names = cluster2["worker"]

                executor = ThreadPoolExecutor(6)

                for id,worker in enumerate(workers):
                    #executor.submit(func, "worker",id)
                    executor.submit(func2, "worker", id,worker_names[id])

                executor.submit(func, "ps", 0)



                params=[]
                epochs=100
                DistTrain = distributed_training.distributed_training(params, file_name)
                #DistTrain.train("ps", 0, epochs)
                DistTrain.train("master", 0, epochs)

                for id,worker in enumerate(workers):
                    ip_port=worker.split(":")
                    port=ip_port[1]
                    executor.submit(clear_port, port, worker_names[id])

                directory = os.path.dirname(os.getcwd())
                model_dir=directory+"/master"

                accuracy=DistTrain.test(model_dir)


                if accuracy==None:
                    to_send="NONE"
                else:
                    to_send=str(accuracy)
                connection.sendall(to_send.encode())


            logger.debug("Sent data")
    except:
        logger.exception("Problem handling request")
    finally:
        logger.debug("Closing socket")

class Server(object):

    def __init__(self, hostname, port,globalIP):
        import logging
        self.logger = logging.getLogger("server")
        self.hostname = hostname
        self.port = port
        self.globalIP = globalIP




    def start(self):
        import logging
        logging.basicConfig(level=logging.DEBUG)


        executor = ThreadPoolExecutor(5)
        self.logger.debug("listening")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(5)




        while True:
            conn, address = self.socket.accept()
            self.logger.debug("Got connection")

            thread = executor.submit(handle, conn, address)
            #process.daemon = True
            #thread.start()
            self.logger.debug("Started process %r", thread)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    globalIP=sys.argv[1]
    ip = sys.argv[2]
    port = sys.argv[3]

    server = Server(ip, int(port),globalIP)


    try:
        logging.info("Listening")
        server.start()
    except:
        logging.exception("Unexpected exception")
    finally:
        logging.info("Shutting down")
        #logging.info("server handles %r operations in %r seconds", str(counter),str(period))


