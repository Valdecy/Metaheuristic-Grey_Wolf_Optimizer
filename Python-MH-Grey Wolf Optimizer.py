############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Grey Wolf Optimizer

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Grey_Wolf_Optimizer, File: Python-MH-Grey Wolf Optimizer.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Grey_Wolf_Optimizer>

############################################################################

# Required Libraries
import pandas as pd
import numpy  as np
import random
import os

# Function: Initialize Variables
def initial_position(pack_size = 5, min_values = [-5,-5], max_values = [5,5]):
    position = pd.DataFrame(np.zeros((pack_size, len(min_values))))
    position['Fitness'] = 0.0
    for i in range(0, pack_size):
        for j in range(0, len(min_values)):
             position.iloc[i,j] = random.uniform(min_values[j], max_values[j])
        position.iloc[i,-1] = target_function(position.iloc[i,0:position.shape[1]-1])
    return position

# Function: Initialize Alpha
def alpha_position(dimension = 2):
    alpha = pd.DataFrame(np.zeros((1, dimension)))
    alpha['Fitness'] = 0.0
    for j in range(0, dimension):
        alpha.iloc[0,j] = 0.0
    alpha.iloc[0,-1] = target_function(alpha.iloc[0,0:alpha.shape[1]-1])
    return alpha

# Function: Initialize Beta
def beta_position(dimension = 2):
    beta = pd.DataFrame(np.zeros((1, dimension)))
    beta['Fitness'] = 0.0
    for j in range(0, dimension):
        beta.iloc[0,j] = 0.0
    beta.iloc[0,-1] = target_function(beta.iloc[0,0:beta.shape[1]-1])
    return beta

# Function: Initialize Delta
def delta_position(dimension = 2):
    delta = pd.DataFrame(np.zeros((1, dimension)))
    delta['Fitness'] = 0.0
    for j in range(0, dimension):
        delta.iloc[0,j] = 0.0
    delta.iloc[0,-1] = target_function(delta.iloc[0,0:delta.shape[1]-1])
    return delta

# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta, min_values = [-5,-5], max_values = [5,5]):
    updated_position = position.copy(deep = True)
    for i in range(0, position.shape[0]):
        if (updated_position.iloc[i,-1] < alpha.iloc[0,-1]):
            for j in range(0, updated_position.shape[1]):
                alpha.iloc[0,j] = updated_position.iloc[i,j]
        if (updated_position.iloc[i,-1] > alpha.iloc[0,-1] and updated_position.iloc[i,-1] < beta.iloc[0,-1]):
            for j in range(0, updated_position.shape[1]):
                beta.iloc[0,j] = updated_position.iloc[i,j]
        if (updated_position.iloc[i,-1] > alpha.iloc[0,-1] and updated_position.iloc[i,-1] > beta.iloc[0,-1]  and updated_position.iloc[i,-1] < delta.iloc[0,-1]):
            for j in range(0, updated_position.shape[1]):
                delta.iloc[0,j] = updated_position.iloc[i,j] 
    return alpha, beta, delta

# Function: Updtade Position
def update_position(position, alpha, beta, delta, a_linear_component = 2, min_values = [-5,-5], max_values = [5,5]):
    updated_position = position.copy(deep = True)
    
    for i in range(0, updated_position.shape[0]):
        for j in range (0, len(min_values)):   
            r1_alpha = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_alpha = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_alpha = 2*a_linear_component*r1_alpha - a_linear_component
            c_alpha = 2*r2_alpha
            
            distance_alpha = abs(c_alpha*alpha.iloc[0,j] - position.iloc[i,j]) 
            x1 = alpha.iloc[0,j] - a_alpha*distance_alpha
        
            r1_beta = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_beta = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_beta = 2*a_linear_component*r1_beta - a_linear_component
            c_beta = 2*r2_beta   
            
            distance_beta = abs(c_beta*beta.iloc[0,j] - position.iloc[i,j]) 
            x2 = beta.iloc[0,j] - a_beta*distance_beta
                             
            r1_delta = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2_delta = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            a_delta = 2*a_linear_component*r1_delta - a_linear_component
            c_delta = 2*r2_delta   
            
            distance_delta = abs(c_delta*delta.iloc[0,j] - position.iloc[i,j]) 
            x3 = delta.iloc[0,j] - a_delta*distance_delta           
                        
            updated_position.iloc[i,j] = (x1 + x2 + x3)/3
            if (updated_position.iloc[i,j] > max_values[j]):
                updated_position.iloc[i,j] = max_values[j]
            elif (updated_position.iloc[i,j] < min_values[j]):
                updated_position.iloc[i,j] = min_values[j]        

        updated_position.iloc[i,-1] = target_function(updated_position.iloc[i,0:updated_position.shape[1]-1])

    return updated_position

# GWO Function
def grey_wolf_optimizer(pack_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50):    
    count = 0
    position = initial_position(pack_size = pack_size, min_values = min_values, max_values = max_values)
    alpha = alpha_position(dimension = len(min_values))
    beta = beta_position(dimension = len(min_values))
    delta = delta_position(dimension = len(min_values))

    while (count <= iterations):
        
        print("Iteration = ", count)
        
        a_linear_component = 2 - count*(2/iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta, min_values = min_values, max_values = max_values)
        position = update_position(position, alpha, beta, delta, a_linear_component = a_linear_component, min_values = min_values, max_values = max_values)
        
        count = count + 1 
        
    print(position.iloc[position['Fitness'].idxmin(),:].copy(deep = True))    
    return position.iloc[position['Fitness'].idxmin(),:].copy(deep = True)

######################## Part 1 - Usage ####################################

# Function to be Minimized. Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def target_function (variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

gwo = grey_wolf_optimizer(pack_size = 15, min_values = [-5,-5], max_values = [5,5], iterations = 100)
