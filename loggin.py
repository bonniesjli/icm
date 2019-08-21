import csv
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

########## TO DO's ##########
# - seed plotters

class Logging():
    """
    Log data into appropriate files
    """
    def __init__(self, run):
        self.run = run
        self.variable_names = {}

    def create(self, variable_name):
        """Create a variable name to be logged"""
        for name in self.variable_names:
            if variable_name == name:
                raise ValueError("variable name already exist")
                pass
        self.variable_names[variable_name] = Variable(variable_name) 
        
        
    def log(self, variable_name, data, t_step = None):
        """Log data under given variable name"""
        if t_step is None:
            self.variable_names[variable_name].add(data)
        else: 
            self.variable_names[variable_name].step(data, t_step)        
        
    def save(self, variable_name):
        """Save data under variable name"""
        if len(self.variable_names[variable_name].t) == 0:
            file_name = str(self.run) + str("/") + str(variable_name) + str(".csv")
            print (file_name)
            df = pd.DataFrame(self.variable_names[variable_name].data)
            df.to_csv(file_name, index = False)
        else: 
            d = {'score': self.variable_names[variable_name].data, 't': self.variable_names[variable_name].t}
            file_name = str(self.run) + str("/") + str(variable_name) + str(".csv")
            print (file_name)
            df = pd.DataFrame(d)
            df.to_csv(file_name, index = False)
        
    def save_data(self):
        """Save all data"""
        for variable_name in self.variable_names:
            self.save(variable_name)
            
    def visualize(self, variable_name, avg = False):
        """Visualize data under variable name"""
        file_name = str(self.run) + str("/") + str(variable_name) + str(".png")
        if avg: 
            data = []
            window = deque(maxlen = 100)
            for p in self.variable_names[variable_name].data:
                window.append(p)
                data.append(np.mean(window))
                
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            plt.plot(np.arange(len(data)), data)
            plt.ylabel(str(variable_name))
            plt.xlabel('Steps')    
            plt.savefig(file_name)
        else:
            if len(self.variable_names[variable_name].t) == 0:
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                plt.plot(np.arange(len(self.variable_names[variable_name].data)), self.variable_names[variable_name].data)
                plt.ylabel(str(variable_name))
                plt.xlabel('Steps')
                plt.savefig(file_name)
            else: 
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                plt.plot(self.variable_names[variable_name].t, self.variable_names[variable_name].data)
                plt.ylabel(str(variable_name))
                plt.xlabel('Steps')
                plt.savefig(file_name)
                
    def data(self, variable_name):
        """Return data under variable name"""
        return self.variable_names[variable_name].data
    
    def mean(self, variable_name, length = 100):
        """Return last mean of length under variable name"""
        return np.mean(self.variable_names[variable_name].data[-length:])
        
        
class Variable():
    def __init__(self, name):
        self.name = name
        self.data = []
        self.t = []
    
    def add(self, data_point):
        self.data.append(data_point)
     
    def step(self, data_point, t_step):
        self.data.append(data_point)
        self.t.append(t_step)