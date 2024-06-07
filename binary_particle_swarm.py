import numpy as np
import gurobipy as gp
from gurobipy import *
import pandas as pd
import random
import time
import csv

def constraints(particle):
    """
    Calculate the objective function and penalty for a given particle configuration.

    Parameters:
    particle (list or array): A list or array representing the particle configuration.

    Returns:
    float: The objective function value with penalty for the given particle configuration.
    """
    data_path = "Data.xlsx"
    try:
        df_1 = pd.read_excel(data_path, sheet_name='Q')
        clist = df_1.values.tolist()
        service_value = clist
    except Exception as e:
        print(f"Error reading sheet 'Q' from Data.xlsx: {e}")
        return None

    try:
        df_2 = pd.read_excel(data_path, sheet_name='w')
        weight = list(df_2['weight'])
    except Exception as e:
        print(f"Error reading sheet 'w' from Data.xlsx: {e}")
        return None

    try:
        df_3 = pd.read_excel(data_path, sheet_name='pv')
        population = list(df_3['Population'])
    except Exception as e:
        print(f"Error reading sheet 'pv' from Data.xlsx: {e}")
        return None

    particle = list(particle)
    capacity = 25000
    dc_capacity = 65000
    I = range(25)
    J = range(11)

    x_ = np.zeros((25, 11))
    y_ = [0 for i in range(25)]
    k = 0
    for i in I:
        for j in J:
            x_[i, j] = particle[k]
            k += 1

    dc = dc_capacity
    C = capacity
    w = weight 
    Q = service_value
    pv = population 

    penalty = 0
    for j in J:
        total = 0
        for i in I:
            total += x_[i, j] * C
        if total - pv[j] < 0:
            penalty += 25

    for i in I:
        total = 0
        for j in J:
            total += x_[i, j] * pv[j]
        if total - dc > 0:
            penalty += 25
    for i in I:
        for j in J:
            if x_[i, j] - Q[i][j] > 0:
                penalty += 25

    for i in I:
        for j in J:
            if x_[i, j] == 1:
                y_[i] = 1
                break

    ori = 0
    for i in I:
        ori += w[i] * y_[i]
    orig_obj = ori
    new_obj = orig_obj + penalty

    if penalty == 0:
        print(new_obj, orig_obj)
        
    return new_obj

def sigmo(x):
    """
    Sigmoid function.

    Parameters:
    x (float): The input value.

    Returns:
    float: The sigmoid of the input value.
    """
    return 1 / (1 + np.exp(-x))

def random_initial(population, dimension):
    """
    Generate a random initial population of particles.

    Parameters:
    population (int): The number of particles in the population.
    dimension (int): The dimensionality of each particle.

    Returns:
    array: A numpy array representing the initial population of particles.
    """
    particles = np.array([[1 if random.uniform(0, 1) > 0.5 else 0 for i in range(dimension)] for i in range(population)])
    return particles

def random_initial_2(population, dimension):
    """
    Generate a random initial population of particles based on service values.

    Parameters:
    population (int): The number of particles in the population.
    dimension (int): The dimensionality of each particle.

    Returns:
    array: A numpy array representing the initial population of particles.
    """
    try:
        df_1 = pd.read_excel('Data.xlsx', sheet_name='Q')
        clist = df_1.values.tolist()
        service_val = clist
    except Exception as e:
        print(f"Error reading sheet 'Q' from Data.xlsx: {e}")
        return None

    total_service = 0
    for i in range(25):
        for j in range(11):
            if service_val[i][j] == 1:
                total_service += 1
    k = 0
    zero_list = []
    for i in range(25):
        for j in range(11):
            if service_val[i][j] == 0:
                zero_list.append(k)
            k += 1
    parts = []
    for i in range(population):
        n = random.randint(10, 40)
        first_part = [1 for i in range(n)]
        second_part = [0 for i in range(total_service - n)]
        second_part.extend(first_part)
        random.shuffle(second_part)
        for i in zero_list:
            second_part.insert(i, 0)
        parts.append(second_part)
    return np.array(parts)

def initial_velocity(population, dimension, vmax=4, vmin=-4):
    """
    Generate initial velocities for a population of particles.

    Parameters:
    population (int): The number of particles in the population.
    dimension (int): The dimensionality of each particle.
    vmax (float): The maximum velocity.
    vmin (float): The minimum velocity.

    Returns:
    array: A numpy array representing the initial velocities of the particles.
    """
    velocity = np.array([[vmin + (vmax - vmin) * np.random.uniform(0, 1) for i in range(dimension)] for i in range(population)])
    return velocity

def update_velocity(particle, velocity, pbest, gbest, vmax=4, vmin=-4):
    """
    Update the velocity of a particle.

    Parameters:
    particle (array): The current particle configuration.
    velocity (array): The current velocity of the particle.
    pbest (array): The best position of the particle.
    gbest (array): The best global position.
    vmax (float): The maximum velocity.
    vmin (float): The minimum velocity.

    Returns:
    array: The updated velocity of the particle.
    """
    new_velocity = np.array([0.0 for i in range(len(particle))])
    delta_velocity = np.array([0.0 for i in range(len(particle))])
    r1 = np.random.uniform(0, 1)
    r2 = np.random.uniform(0, 1)
    w = 1.2
    c1 = 2
    c2 = 2

    for i in range(len(particle)):
        delta_velocity[i] = c1 * r1 * (pbest[i] - particle[i]) + c2 * r2 * (gbest[i] - particle[i])
        new_velocity[i] = velocity[i] + delta_velocity[i]

        if new_velocity[i] > vmax:
            new_velocity[i] = vmax
        elif new_velocity[i] < vmin:
            new_velocity[i] = vmin

    return new_velocity

def update_dimensions(particle, velocity):
    """
    Update the dimensions (position) of a particle.

    Parameters:
    particle (array): The current particle configuration.
    velocity (array): The current velocity of the particle.

    Returns:
    array: The updated particle configuration.
    """
    for i in range(len(particle)):
        if sigmo(velocity[i]) > np.random.uniform(0, 1):
            particle[i] = 1
        else:
            particle[i] = 0
    return particle

def particle_swarm(func, population, iterations, dimension):
    """
    Perform Particle Swarm Optimization (PSO).

    Parameters:
    func (function): The objective function to optimize.
    population (int): The number of particles in the swarm.
    iterations (int): The number of iterations to run the PSO.
    dimension (int): The dimensionality of each particle.

    Returns:
    tuple: A tuple containing the list of particles, the best particle position, and the solution time.
    """
    start_time = time.time()
    particles = random_initial_2(population, dimension)
    velocity = initial_velocity(population, dimension)
    pbest_positions = particles.copy()
    pbest_function = [func(p) for p in pbest_positions]
    best_10 = sorted(pbest_function)
    print(best_10[:10])
    gbest_index = np.argmin(pbest_function)
    gbest_position = pbest_positions[gbest_index]
    gbest_function = np.min(pbest_function)
    particle_lst = []

    for i in particles:
        particle_lst.append(i.copy())

    for i in range(iterations):
        for j in range(population):
            velocity[j] = update_velocity(particles[j], velocity[j], pbest_positions[j], gbest_position)
            particles[j] = update_dimensions(particles[j], velocity[j])
            if func(particles[j]) < pbest_function[j]:
                pbest_positions[j] = particles[j].copy()
                pbest_function[j] = func(pbest_positions[j])

            if func(particles[j]) < gbest_function:
                gbest_position = particles[j].copy()
                gbest_function = func(gbest_position)

            particle_lst.append(particles[j].copy())

    solution_time = time.time() - start_time
    return particle_lst, gbest_position, solution_time

s, b, soltime = particle_swarm(constraints, 550, 200, 275)
print("best", constraints(b))
print("solution time:", soltime)
