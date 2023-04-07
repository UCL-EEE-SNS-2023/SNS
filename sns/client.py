import socket
import pickle

def client(message):
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.connect(('127.0.0.1',1234))
            s.sendall(bytes(message,'utf-8'))
            data = s.recv(1024)
            # unpack to get a list of int value
            feedback_list = pickle.loads(data)
            print(feedback_list)
            print('Received:',repr(data))
            s.close()

            return feedback_list
