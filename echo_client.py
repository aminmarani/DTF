import socket
import time
import random

HOST = '3.19.32.246'#'128.180.111.80'#"13.59.40.136"    # The remote host
PORT = 50007              # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
print(s)
id = random.randint(0,100)
for i in range(100):
    s.sendall(str('Hello, world,')+str(i)+','+str(id))
    time.sleep(0.1)
#data = s.recv(1024)
s.close()
#print('Received', repr(data))