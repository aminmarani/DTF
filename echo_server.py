import socket

HOST = '172.31.38.21' #'128.180.111.80'#"172.31.39.90"   #private IP for server              # Symbolic name meaning all available interfaces
#HOST = 'DESKTOP-LSM09S6'
PORT = 50007              # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()
print('Connected by', addr)
while 1:
    data = conn.recv(1024)
    print(data)
    if not data: break
    #conn.sendall(data)
#conn.sendall(str('fin'))
conn.close()