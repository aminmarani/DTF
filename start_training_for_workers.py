import ast
import json
import os
import pickle
import sys

import tensorflow as tf
from sklearn.metrics import accuracy_score

tf.logging.set_verbosity(tf.logging.INFO)

import model_definition
import distributed_training

server_role = sys.argv[1]
server_number = sys.argv[2]
epochs=100
params=[]
cluster_info="distributed_servers"
DistTrain=distributed_training.distributed_training(params,cluster_info)
DistTrain.train(server_role,int(server_number),epochs)