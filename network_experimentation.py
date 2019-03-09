#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:02:22 2019

@author: thomas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bias(array):
    array = np.array(array)
    if len(array.shape) == 1:
        return np.append(array, [1])
    return np.vstack((array, np.ones((1, array.shape[1]))))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Network:
    def __init__(self, layers, learning_rate=0.1, mass=0):
        self.weights = []
        self.delta = []
        last_layer = None
        for layer in layers:
            if last_layer != None:
                self.weights.append(np.random.rand(layer, last_layer + 1) - 0.5)
                self.delta.append(np.zeros((layer, last_layer + 1)))
            last_layer = layer
        self.eta = learning_rate
        self.mass = mass
        
    def forward(self, inputs):
        data = inputs
        for layer in self.weights:
            data = sigmoid(layer @ bias(data))
        return data
    
    def back(self, inputs, targets):
        data = [inputs]
        for layer in self.weights:
            data.append(sigmoid(layer @ bias(data[-1])))
        error = [x * 0 for x in data]
        error[-1] = -(targets - data[-1]) * (data[-1]) * (1 - data[-1])
        for i in range(len(data) - 2, -1, -1):
            error[i] = (np.transpose(self.weights[i]) @ error[i + 1])
            error[i] = np.delete(error[i], -1)
            error[i] *= (data[i]) * (1 - data[i])
            self.weights[i] -= np.outer(error[i + 1], bias(data[i])) * self.eta
        #print(error)
    
    def back_batch(self, inputs, targets):
        data = [np.transpose(inputs)]
        for layer in self.weights:
            data.append(sigmoid(layer @ bias(data[-1])))
        error = [0 for x in data]
        error[-1] = -(np.transpose(targets) - data[-1]) * (data[-1]) * (1 - data[-1])
        for i in range(len(data) - 2, -1, -1):
            error[i] = (np.transpose(self.weights[i]) @ error[i + 1])
            error[i] = np.delete(error[i], -1, axis=0)
            error[i] *= (data[i]) * (1 - data[i])
            self.delta[i] *= self.mass
            self.delta[i] += (error[i + 1] @ np.transpose(bias(data[i]))) * self.eta
            self.weights[i] -= self.delta[i]


def load_car():
    df = pd.read_csv("car.csv", header=None, names=["buying","maint","doors",
                                                    "persons","lug_boot",
                                                    "safety","class"])
    
    df["buying"] = df["buying"].map({"low":0, "med":1, "high":2, "vhigh":3})
    df["maint"] = df["maint"].map({"low":0, "med":1, "high":2, "vhigh":3})
    df["doors"] = df["doors"].map({"2":2, "3":3, "4":4, "5more":5})
    df["persons"] = df["persons"].map({"2":2, "4":4, "more":5})
    df["lug_boot"] = df["lug_boot"].map({"small":0, "med":1, "big":2})
    df["safety"] = df["safety"].map({"low":0, "med":1, "high":2})
    
    df["unacc"] = df["class"].map(lambda x: 1 if x == "unacc" else 0)
    df["acc"] = df["class"].map(lambda x: 1 if x == "acc" else 0)
    df["good"] = df["class"].map(lambda x: 1 if x == "good" else 0)
    df["vgood"] = df["class"].map(lambda x: 1 if x == "vgood" else 0)
    df = df.drop("class", 1)
    
    n = np.array(df.sample(frac=1))
    
    return n[:,:-4], n[:,-4:]

np.random.seed(1)

data, targ = load_car()
data = data / np.std(data, axis=0)
split = (len(data) * 7) // 10

def test_network_paramiters(prams, tries=2):
    acc = 0
    for s in range(tries):
        np.random.seed(s)
        net = Network(*prams)
        for i in range(4000):
            net.back_batch(data[:split], targ[:split])
        acc += np.mean([np.argmax(net.forward(data[i])) == np.argmax(targ[i]) for i in range(split,len(targ))]) * 100
    
    return acc/tries

networks = []

for eta in [0.02, 0.01, 0.005, 0.002]:
    for m in [0, 0.1, 0.3, 0.5]:
        for a in [5,8,10,15]:
            print("eta = {}, m = {}, a = {}".format(eta, m, a))
            n = ([6, a, 4], eta, m)
            networks.append((n, test_network_paramiters(n)))
            for b in [5,8,10,15]:
                n = ([6, a, b, 4], eta, m)
                networks.append((n, test_network_paramiters(n)))