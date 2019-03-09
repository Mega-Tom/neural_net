#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 22:14:28 2019

@author: thomas
"""

import numpy as np
import pandas as pd

def bias(array):
    return np.append(array, [1])

class Percepitron:
    def __init__(self, input, output, learning_rate=0.1):
        self.weights = np.random.rand(output, input + 1) - 0.5
        self.eta = learning_rate
        
    def forward(self, inputs):
        return np.where((self.weights @ bias(inputs)) > 0, 1, 0)
    
    def back(self, inputs, targets):
        error = self.forward(inputs) - targets
        self.weights -= np.outer(error, bias(inputs)) * self.eta
        
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

data, targ = load_car()

data = data / np.std(data, axis=0)

p = Percepitron(data.shape[1], targ.shape[1], 0.05)
for _ in range(200):
    for i in range(len(data)//2):
        p.back(data[i], targ[i])

print(np.mean([p.forward(data[i]) == targ[i] for i in range(len(data)//2, len(data))])*100)