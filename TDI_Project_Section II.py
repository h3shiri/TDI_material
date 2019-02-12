
# coding: utf-8

# ## This notebook contains calculations for the TDI challenge second part

# ### Most of the stimates were calculated with monte carlo simulations in mind 

# In[92]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras # for machine learning models
import scipy # linear models..etc
import random # for random choices..etc
import statistics as st # Python satistics module


#Taking both time inputs as strings and returning float
import datetime
#We import the required libraries for data exploration and visualization.
import matplotlib.pyplot as plt
import os
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir("."))


# In[93]:


#Model for simulating the values of the cars.
# we have range of positions form 0 to N-1.
# We have cars starting from positions 0 to M-1.
# The number of moves is calculated by T.


# In[94]:


#Representing a coordinate in our board structure
class Coordinate:
    def __init__(self, id):
        self.id = id
        self.availability = True
        self.car = None
        self.nextCoordinate = None
        self.previousCoordinate = None
        
    def getNext(self):
        return self.nextCoordinate
    
    def getPrevious(self):
        return self.previousCoordinate
    
    def getAvailability(self):
        return self.availability
    
    def getId(self):
        return self.id
    
    def setNextCoordinate(self, cor):
        self.nextCoordinate = cor
        
    def setPreCoordinate(self, cor):
        self.previousCoordinate = cor
    
    def setAsTaken(self):
        self.availability = False
    
    def setAsFree(self):
        self.availability = True
        
    def getCar(self):
        return self.car
    
    def setCar(self, car):
        self.car = car
        
    def switchCarToRunning(self):
        if(self.car != None):
            self.car.setCanMove()


# In[95]:


#Representing a car object in our data
class Car:
    def __init__(self, id, pos):
        self.id = id
        self.canMove = False
        self.position = pos
        
    def getId(self):
        return self.id
    
    def getMoveStatus(self):
        return self.canMove
    
    # Moving the car along the board
    def move(self):
        if(self.position.getNext().getAvailability()):
            prevPosition = self.position
            self.position.getNext().setCar(self)
            self.position.getNext().setAsTaken()
            self.position = prevPosition.getNext()
            
            #Updating availabilty options for car behind me
            prevPosition.getPrevious().switchCarToRunning()
            # Updating the previous position
            prevPosition.setAsFree()
            prevPosition.setCar(None)
    
    # Flag wether the car may move forward
    def setCanMove(self):
        self.canMove = True
        
    def setCannotMove(self):
        self.canMove = False
        
    
    def __repr__(self):
        return "A Car with ID : {0}, and position : {1}".format(self.id, self.position.getId())
            
    


# In[96]:


import statistics as st
st.mean([2,3,4,5])


# In[125]:


#Class for all these objects
class roundBoard :
    
    #Setting the coordinates and the like..etc
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.positions = [Coordinate(i) for i in range(N)]
        # Setting the pointers correctly
        for index in range(N):
            nextIndex = (index + 1) % N
            preIndex = (index - 1) % N
            self.positions[index].setNextCoordinate(self.positions[nextIndex])
            self.positions[index].setPreCoordinate(self.positions[preIndex])
            
        # initialising the cars
        for index in range(M):
            self.positions[index].setAsTaken()
            self.positions[index].setCar(Car(index, self.positions[index]))
        #Flagging the last car as the one that can move 
        self.positions[M-1].car.setCanMove()
            
    #Fetching the mean value of the car locations
    def calculateExpected_A_Value(self):
        tempArr = []
        for pos in self.positions:
            if(not pos.availability):
                tempArr.append(pos.getId())
        return st.mean(tempArr)
    
    #fetching Standard Deviation
    def calculateStd(self):
        tempArr = []
        for pos in self.positions:
            if(not pos.availability):
                tempArr.append(pos.getId())
        return st.stdev(tempArr)
    
    def printCarLocations(self):
        res = ""
        for pos in self.positions :
            if(pos.availability):
                res += 'o'
            else:
                res+= 'X'
        res += '\n' + '_'*self.N
        print(res)
        
    def printExtendedInfo(self):
        res = ""
        for pos in self.positions :
            if(pos.availability):
                res += 'o'
            else:
                res+= str(pos.getCar().getId())
        res += '\n' + '_'*self.N
        print(res)
    
    def getPotentialCarsToMove(self):
        res = []
        for pos in self.positions:
            if(not pos.getAvailability()):
                if(pos.getCar().getMoveStatus()):
                    res.append(pos.getId())
        return res
    
    def moveOneOfTheCars(self):
        potential_pos_ids = self.getPotentialCarsToMove()
        targetPosIndex = random.choice(potential_pos_ids)
        self.positions[targetPosIndex].getCar().move()


# In[126]:


random.choice([2])


# In[128]:


b1 = roundBoard(10,5)
b1.printExtendedInfo()
for _ in range(20):
    b1.moveOneOfTheCars()
    b1.printExtendedInfo()


# In[121]:


b1 = roundBoard(10,5)
b1.printCarLocations()
for _ in range(20):
    b1.moveOneOfTheCars()

b1.printCarLocations()


# In[131]:


class Game :
    
    def __init__(self, N, M, T):
        self.board = roundBoard(N,M)
        self.turns = T
    
    # Shuffle the board by one step
    def shuffle(self):
        if (self.turns > 0):
            self.board.moveOneOfTheCars()
            self.turns -= 1
    
    def get_expected_A(self):
        while(self.turns):
            self.shuffle()
        return self.board.calculateExpected_A_Value()
    
    def getStd_A(self):
        while(self.turns):
            self.shuffle()
        return self.board.calculateStd()
    
    def play(self):
        while(self.turns):
            self.shuffle()
        print("The results are Exp A : {0}, and std : {1}".format(self.board.calculateExpected_A_Value(), self.board.calculateStd()))
    


# In[143]:


estimation_cycles = 200000
N = 10
M = 5
T = 20
avg_res = 0
avg_std = 0
for _ in range(estimation_cycles):
    g1 = Game(N,M,T)
    avg_res += g1.get_expected_A()
    avg_std += g1.getStd_A()
print("exp A : {0}".format(np.round(avg_res/estimation_cycles, 10)))
print("stdDev A : {0}".format(np.round(avg_std/estimation_cycles, 10)))


# In[144]:


for i in range(10, 20, 2):
    print(i)


# In[146]:


Possible_estmation_magnitudes = range(10000, 20000,1000)
temp_vals_for_S = []
estimation_cycles = 200000
N = 10
M = 5
T = 20
for estimation_cycles in Possible_estmation_magnitudes:
    std_val = 0
    for _ in range(estimation_cycles):
        g1 = Game(N,M,T)
        std_val += g1.getStd_A()
    temp_vals_for_S.append(std_val/estimation_cycles)
print("exp S : {0}".format(np.round(st.mean(temp_vals_for_S), 10)))
print("stdDev S : {0}".format(np.round(st.stdev(temp_vals_for_S), 10)))


# In[147]:


estimation_cycles = 200000
N = 25
M = 10
T = 50
avg_res = 0
avg_std = 0
for _ in range(estimation_cycles):
    g1 = Game(N,M,T)
    avg_res += g1.get_expected_A()
    avg_std += g1.getStd_A()
print("exp A : {0}".format(np.round(avg_res/estimation_cycles, 10)))
print("stdDev A : {0}".format(np.round(avg_std/estimation_cycles, 10)))


# In[148]:


Possible_estmation_magnitudes = range(10000, 20000,1000)
temp_vals_for_S = []
estimation_cycles = 200000
N = 25
M = 10
T = 50
for estimation_cycles in Possible_estmation_magnitudes:
    std_val = 0
    for _ in range(estimation_cycles):
        g1 = Game(N,M,T)
        std_val += g1.getStd_A()
    temp_vals_for_S.append(std_val/estimation_cycles)
print("exp S : {0}".format(np.round(st.mean(temp_vals_for_S), 10)))
print("stdDev S : {0}".format(np.round(st.stdev(temp_vals_for_S), 10)))

