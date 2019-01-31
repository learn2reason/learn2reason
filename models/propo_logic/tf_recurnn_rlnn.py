#!/usr/bin/env python
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten


leaves = ['0', '1', 'and', 'or', 'not', '(', ')',\
        '( 0', '( 1', '0 )', '1 )',\
        '0 and', '0 or', '1 and', '1 or',\
        'and 0', 'or 0', 'and 1', 'or 1', 'error!']
erroridx = leaves.index('error!')
ncombs = len(leaves)
nsymbols = 7
nvalue = ncombs
nprece = 4
slim = tf.contrib.slim

class value_NN0(object):

    def __init__(self, init_v, hiddens, nadd, v_epsilon=1e-9):
        self.W_s1 = tf.get_variable("W_s1",[2*ncombs+nadd, hiddens[0]],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s1 = tf.get_variable("b_s1",[hiddens[0],],initializer=tf.random_uniform_initializer(0,0))
        self.W_s2 = tf.get_variable("W_s2",[hiddens[0],2],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s2 = tf.get_variable("b_s2",[2],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(None, 2*ncombs+nadd))
        self.targets = tf.placeholder(tf.float32,  shape=(None, 2))
        
        self.params = [self.W_s1, self.b_s1, self.W_s2, self.b_s2]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_s1) + self.b_s1
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s2) + self.b_s2
        return score

class value_NN1(object):

    def __init__(self, init_v, hiddens, nadd, v_epsilon=1e-9):
        self.W_s1 = tf.get_variable("W_s1",[2*ncombs+nadd, 2],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s1 = tf.get_variable("b_s1",[2,],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(None, 2*ncombs+nadd))
        self.targets = tf.placeholder(tf.float32,  shape=(None, 2))
        
        self.params = [self.W_s1, self.b_s1]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_s1) + self.b_s1
        return score

class value_NN2(object):

    def __init__(self, init_v, hiddens, nadd, v_epsilon=1e-9):
        self.W_s1 = tf.get_variable("W_s1",[2*ncombs+nadd, hiddens[0]],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s1 = tf.get_variable("b_s1",[hiddens[0],],initializer=tf.random_uniform_initializer(0,0))
        self.W_s2 = tf.get_variable("W_s2",[hiddens[0],hiddens[1]],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s2 = tf.get_variable("b_s2",[hiddens[1]],initializer=tf.random_uniform_initializer(0,0))
        self.W_s3 = tf.get_variable("W_s3",[hiddens[1],hiddens[2]],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s3 = tf.get_variable("b_s3",[hiddens[2]],initializer=tf.random_uniform_initializer(0,0))
        self.W_s4 = tf.get_variable("W_s4",[hiddens[2], 2],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s4 = tf.get_variable("b_s4",[2],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(None, 2*ncombs+nadd))
        self.targets = tf.placeholder(tf.float32,  shape=(None, 2))
        
        self.params = [self.W_s1, self.b_s1, self.W_s2, self.b_s2,\
                self.W_s3, self.b_s3, self.W_s4, self.b_s4]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_s1) + self.b_s1
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s2) + self.b_s2
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s3) + self.b_s3
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s4) + self.b_s4
        return score

class value_NN3(object):

    def __init__(self, init_v, hiddens, nadd, v_epsilon=1e-9):
        self.W_s1 = tf.get_variable("W_s1",[2*ncombs+nadd, hiddens[0]],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s1 = tf.get_variable("b_s1",[hiddens[0],],initializer=tf.random_uniform_initializer(0,0))
        self.W_s2 = tf.get_variable("W_s2",[hiddens[0],hiddens[1]],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s2 = tf.get_variable("b_s2",[hiddens[1]],initializer=tf.random_uniform_initializer(0,0))
        self.W_s3 = tf.get_variable("W_s3",[hiddens[1],hiddens[2]],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s3 = tf.get_variable("b_s3",[hiddens[2]],initializer=tf.random_uniform_initializer(0,0))
        self.W_s4 = tf.get_variable("W_s4",[hiddens[2], hiddens[3]],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s4 = tf.get_variable("b_s4",[hiddens[3]],initializer=tf.random_uniform_initializer(0,0))
        self.W_s5 = tf.get_variable("W_s5",[hiddens[3], hiddens[4]],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s5 = tf.get_variable("b_s5",[hiddens[4],],initializer=tf.random_uniform_initializer(0,0))
        self.W_s6 = tf.get_variable("W_s6",[hiddens[4],hiddens[5]],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s6 = tf.get_variable("b_s6",[hiddens[5]],initializer=tf.random_uniform_initializer(0,0))
        self.W_s7 = tf.get_variable("W_s7",[hiddens[5],hiddens[6]],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s7 = tf.get_variable("b_s7",[hiddens[6]],initializer=tf.random_uniform_initializer(0,0))
        self.W_s8 = tf.get_variable("W_s8",[hiddens[6], 2],initializer=tf.contrib.layers.xavier_initializer())
        self.b_s8 = tf.get_variable("b_s8",[2],initializer=tf.random_uniform_initializer(0,0))


        # self.W_s1 = tf.get_variable("W_s1",[2*ncombs+nadd, hiddens[0]],initializer=tf.random_uniform_initializer(-init_v,init_v))
        # self.b_s1 = tf.get_variable("b_s1",[hiddens[0],],initializer=tf.random_uniform_initializer(0,0))
        # self.W_s2 = tf.get_variable("W_s2",[hiddens[0], hiddens[1]],initializer=tf.random_uniform_initializer(-init_v,init_v))
        # self.b_s2 = tf.get_variable("b_s2",[hiddens[1]],initializer=tf.random_uniform_initializer(0,0))
        # self.W_s3 = tf.get_variable("W_s3",[hiddens[1], hiddens[2]],initializer=tf.random_uniform_initializer(-init_v,init_v))
        # self.b_s3 = tf.get_variable("b_s3",[hiddens[2]],initializer=tf.random_uniform_initializer(0,0))
        # self.W_s4 = tf.get_variable("W_s4",[hiddens[2], hiddens[3]],initializer=tf.random_uniform_initializer(-init_v,init_v))
        # self.b_s4 = tf.get_variable("b_s4",[hiddens[3]],initializer=tf.random_uniform_initializer(0,0))
        # self.W_s5 = tf.get_variable("W_s5",[hiddens[3], hiddens[4]],initializer=tf.random_uniform_initializer(-init_v,init_v))
        # self.b_s5 = tf.get_variable("b_s5",[hiddens[4],],initializer=tf.random_uniform_initializer(0,0))
        # self.W_s6 = tf.get_variable("W_s6",[hiddens[4], hiddens[5]],initializer=tf.random_uniform_initializer(-init_v,init_v))
        # self.b_s6 = tf.get_variable("b_s6",[hiddens[5]],initializer=tf.random_uniform_initializer(0,0))
        # self.W_s7 = tf.get_variable("W_s7",[hiddens[5], hiddens[6]],initializer=tf.random_uniform_initializer(-init_v,init_v))
        # self.b_s7 = tf.get_variable("b_s7",[hiddens[6]],initializer=tf.random_uniform_initializer(0,0))
        # self.W_s8 = tf.get_variable("W_s8",[hiddens[6], 2],initializer=tf.random_uniform_initializer(-init_v,init_v))
        # self.b_s8 = tf.get_variable("b_s8",[2],initializer=tf.random_uniform_initializer(0,0))

        self.e_in = tf.placeholder(tf.float32,  shape=(None, 2*ncombs+nadd))
        self.targets = tf.placeholder(tf.float32,  shape=(None, 2))
        
        self.params = [self.W_s1, self.b_s1, self.W_s2, self.b_s2,\
                self.W_s3, self.b_s3, self.W_s4, self.b_s4,\
                self.W_s5, self.b_s5, self.W_s6, self.b_s6,\
                self.W_s7, self.b_s7, self.W_s8, self.b_s8]

    def next_score(self):
        score = tf.matmul(self.e_in, self.W_s1) + self.b_s1
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s2) + self.b_s2
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s3) + self.b_s3
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s4) + self.b_s4
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s5) + self.b_s5
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s6) + self.b_s6
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s7) + self.b_s7
        score = tf.nn.relu(score)
        score = tf.matmul(score, self.W_s8) + self.b_s8
        return score


class value_NN4(object):

    def __init__(self, init_v, hiddens, nadd, v_epsilon=1e-9):
        # self.conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = 0, stddev = init_v))
        # self.conv1_b = tf.Variable(tf.zeros(6))
        # self.conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = 0, stddev = init_v))
        # self.conv2_b = tf.Variable(tf.zeros(16))
        # self.fc1_w   = tf.Variable(tf.truncated_normal(shape = (hiddens[2],120), mean = 0, stddev = init_v))
        # self.fc1_b   = tf.Variable(tf.zeros(120))
        # self.fc2_w   = tf.Variable(tf.truncated_normal(shape = (120,84), mean = 0, stddev = init_v))
        # self.fc2_b   = tf.Variable(tf.zeros(84))
        # self.fc3_w   = tf.Variable(tf.truncated_normal(shape = (84,2), mean = 0 , stddev = init_v))
        # self.fc3_b   = tf.Variable(tf.zeros(2))

        self.conv1_w  = tf.get_variable("conv1_w",[5,5,1,6],initializer=tf.contrib.layers.xavier_initializer())
        self.conv1_b  = tf.get_variable("conv1_b",[6,],initializer=tf.random_uniform_initializer(0,0))
        self.conv2_w  = tf.get_variable("conv2_w",[5,5,6,16],initializer=tf.contrib.layers.xavier_initializer())
        self.conv2_b  = tf.get_variable("conv2_b",[16],initializer=tf.random_uniform_initializer(0,0))
        self.fc1_w    = tf.get_variable("fc1_w",  [hiddens[2],120],initializer=tf.contrib.layers.xavier_initializer())
        self.fc1_b    = tf.get_variable("fc1_b",  [120],initializer=tf.random_uniform_initializer(0,0))
        self.fc2_w    = tf.get_variable("fc2_w",  [120, 84],initializer=tf.contrib.layers.xavier_initializer())
        self.fc2_b    = tf.get_variable("fc2_b",  [84],initializer=tf.random_uniform_initializer(0,0))
        self.fc3_w    = tf.get_variable("fc3_w",  [84, 2],initializer=tf.contrib.layers.xavier_initializer())
        self.fc3_b    = tf.get_variable("fc3_b",  [2,],initializer=tf.random_uniform_initializer(0,0))
        
        self.e_in = tf.placeholder(tf.float32,  shape=(None, 2*ncombs+nadd))
        self.targets = tf.placeholder(tf.float32,  shape=(None, 2))
        
        self.params = [self.conv1_w, self.conv1_b, self.conv2_w, self.conv2_b, \
	self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b, self.fc3_w, self.fc3_b]
        self.hiddens = hiddens

        self.W_t = np.zeros((self.hiddens[0]+self.hiddens[1], self.hiddens[0]*self.hiddens[1]), dtype=np.float32)
        self.b_t = np.zeros((self.hiddens[0]*self.hiddens[1], ), dtype=np.float32)
        for i in range(self.hiddens[0]):
            for j in range(self.hiddens[1]):
                self.W_t[i][i*self.hiddens[1]+j] = 1
                self.W_t[j+self.hiddens[0]][i*self.hiddens[1]+j] = 1
        self.b_t -= 1
        self.W_t = tf.constant(self.W_t)
        self.b_t = tf.constant(self.b_t)

    def next_score(self):
        # e_in = tf.reshape(self.e_in, (-1, self.hiddens[0], self.hiddens[1], 1))

        e_in = tf.matmul(self.e_in, self.W_t) + self.b_t
        e_in = tf.nn.relu(e_in)
        e_in = tf.reshape(e_in, (-1, self.hiddens[0], self.hiddens[1], 1))

        conv1 = tf.nn.conv2d(e_in, self.conv1_w, strides = [1,1,1,1], padding = 'VALID') + self.conv1_b 
        conv1 = tf.nn.relu(conv1)
        pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
        conv2 = tf.nn.conv2d(pool_1, self.conv2_w, strides = [1,1,1,1], padding = 'VALID') + self.conv2_b
        conv2 = tf.nn.relu(conv2)
        pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
        fc1 = flatten(pool_2)
        fc1 = tf.matmul(fc1,self.fc1_w) + self.fc1_b
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.matmul(fc1,self.fc2_w) + self.fc2_b
        fc2 = tf.nn.relu(fc2)
        logits = tf.matmul(fc2, self.fc3_w) + self.fc3_b
        return logits

