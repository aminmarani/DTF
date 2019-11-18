import numpy as np
import socket
from multiprocessing import Pool
import multiprocessing.pool
from collections import Counter
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from random import randint
import random
import threading
import time
import itertools #making combination of sets
import threading

#ssh -i launch-wizard-1.pem ec2-user@ec2-13-59-40-136.us-east-2.compute.amazonaws.com


#Client class --> we made this class so we can shared memory through it
class Client:
    #parameters inititiation
    def __init__(self,server_avail,server_list):
        self.global_variable = 0
        self.server_list = server_list
        self.server_avail = server_avail
        self.acc = [0,0,0,0] #zero at the begining for 50,100,150,200 epoches
        self.best_model = []
        self.writing_lock = threading.Lock()

    #handle function
    def handle(self,host, port,params,server_id,server_list):
        #------------AMIN-------------
        #for now, I considered there is no crash at connection --> later I will add an except to handle any crash at connect and
        #return current set of parameters to the list for later process 
        model_acc = []
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            for i in range(4):
                #make the dictionary and message to send for node with id = server_node
                d={'nb1':params[0],'nb2':params[1],'nb3':params[2],'epoch':50}
                d_string=str(d)
                to_send="TRAIN "+d_string+'+1+'+str(server_id)
                #send the message to the node
                s.send(to_send.encode('ascii'))
                #wait to recieve a response
                data = s.recv(1024)
                response = float(data.decode('ascii'))
                #if the acc is bigger then continue and if it is last epoch then store the whole model
                if(response> self.acc[i]):
                    #save current accuracy for later storage
                    model_acc.append(response)
                    if(i == 3): #if it is the last epoch/message --> save the best model then!
                        current_best = self.best_model

                        #while any other thread are writing, we have to wait
                        while(self.writing_lock.locked()):
                            time.sleep(0.1)
                        #lock to write
                        self.writing_lock.acquire()
                        self.acc = model_acc
                        self.best_model = [server_id,params[0],params[1],params[2]]
                        #release the lock
                        self.writing_lock.release()
                        
                        #remove previous best model data folder --> send a message for that
                        s.close() #close current connection
                        #open a new connection
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.connect((server_list[current_best[0]], port))
                        d={'nb1':current_best[1],'nb2':current_best[2],'nb3':current_best[3],'epoch':50}
                        d_string=str(d)
                        to_send="DELETE "+d_string+'+1+'+str(current_best[0])
                        s.send(to_send.encode('ascii'))
                        s.close()
                else: #send delete to the node and go on!
                    to_send="DELETE"+d_string+'+1+'+str(server_id)
                    #send the message to the node
                    s.send(to_send.encode('ascii'))
                    #exit the while loop to end this thread
                    break
                    #----------AMIN--------
                    #server may be 1)shut_down, then we should not care and we can carry on to next operations 2)not getting the message, 
                    #then we should wait for a message to be recved and says something like "ACCEPT" or keep sending message  --> I am not sure if it is neccassary or not!
                print(to_send,'...')
        except:
            #close the connection and make the current server available
            s.close() 
            self.server_avail[server_id] = 1
            #if the lock has not been released, just release it
            if(self.writing_lock.locked()):
                self.writing_lock.release()

        s.close()
        #At the end set this server available
        self.server_avail[server_id] = 1


#running part of codes

#making combination of filters
nb = [[5,10,15],[5,10,15],[12,15,17,20]]
filters = list(itertools.product(*nb))

#reading list of servers from a file
file = open('server_list', 'r')  #read the address file
server_list = []
server_avail = [] #a vector to see if the server is available or not
#add nodes address to nodes_list without a key for now
for lines in file:
    result = lines.find('\n') #remove \n if ther is any
    if(result>-1):
        lines = lines[0:result]
    server_list.append(lines)
    server_avail.append(1)


#host and port
#host = '128.180.111.80'
port = 5000 #Define connection port

#define number of active threads
executor = ThreadPoolExecutor(5)
#make an object of Client Class
cl = Client(server_avail,server_list)


#this is the main body of client...we continue checking available server and at the same time recieving data from servers
#1-when we find an available server we send parameters set and ask to do the training --> set the server to UNAVAILABLE
#2-if the returns acc is less than what we stored before, then we send delete to server and wait for confirmation --> set the node to AVAILABLE
#3-if the server returns acc greater than any node before, we send next call for epoch(50,100,150,200) 
while(len(filters)>0):
    #if there is at least one server available, sumbit a thread to it
    if(sum(cl.server_avail)>0):
        print(cl.best_model)
        print(cl.acc)
        #set server_avail to zero
        index = cl.server_avail.index(1); cl.server_avail[index] = 0
        #get the topest set of params and submit a thread to run it
        params = filters.pop()
        thread = executor.submit(cl.handle, server_list[index], port,params,index,server_list)
    else: #otherwis sleep for 5 seconds and check again if there is an empty client
        time.sleep(5)

#wait till all threads finish their jobs
while(sum(cl.server_avail)< len(cl.server_avail)):
    time.sleep(2)

print('finished  ',cl.best_model)
