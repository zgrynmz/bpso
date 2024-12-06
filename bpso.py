import numpy as np
import gurobipy as gp
from gurobipy import *
import pandas as pd
import random
import time
import csv

# Preload data for efficiency
data_path = "Data.xlsx"
try:
    df_1 = pd.read_excel(data_path, sheet_name='Q')
    service_value = df_1.values  # Use numpy array for faster access
    df_2 = pd.read_excel(data_path, sheet_name='w')
    weight = df_2['weight'].values  # Use numpy array for faster access
    df_3 = pd.read_excel(data_path, sheet_name='pv')
    population = df_3['Population'].values  # Use numpy array for faster access
except Exception as e:
    print(f"Error reading data from Data.xlsx: {e}")
    exit()

capacity = 25000
dc_capacity = 65000
I = range(25)
J = range(11)

def constraints(particle):
    """
    Calculate the objective function and penalty for a given particle configuration.

    Parameters:
    particle (list or array): A list or array representing the particle configuration.

    Returns:
    float: The objective function value with penalty for the given particle configuration.
    """
    global service_value, weight, population, capacity, dc_capacity, I, J

    x_ = np.reshape(particle, (25, 11))  # Efficiently reshape particle
    y_ = np.zeros(25, dtype=int)  # Pre-allocate numpy array for efficiency
    
    penalty = 0
    
    # Calculate penalties using vectorized operations for efficiency
    for j in J:
        total = np.sum(x_[:, j] * capacity)
        if total - population[j] < 0:
            penalty += 25

    for i in I:
        total = np.sum(x_[i, :] * population)
        if total - dc_capacity > 0:
            penalty += 25

    for i in I:
        for j in J:
            if x_[i, j] - service_value[i, j] > 0:
                penalty += 25

    # Find the first assigned service for each facility
    for i in I:
        for j in J:
            if x_[i, j] == 1:
                y_[i] = 1
                break

    # Calculate objective function with vectorized operations for efficiency
    orig_obj = np.sum(weight * y_)
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

def random_initial_2(population, dimension):
    """
    Generate a random initial population of particles based on service values.

    Parameters:
    population (int): The number of particles in the population.
    dimension (int): The dimensionality of each particle.

    Returns:
    array: A numpy array representing the initial population of particles.
    """
    global service_value
    
    total_service = np.sum(service_value)
    k = 0
    zero_list = []
    for i in range(25):
        for j in range(11):
            if service_value[i, j] == 0:
                zero_list.append(k)
            k += 1

    parts = []
    for i in range(population):
        n = random.randint(10, 40)
        first_part = np.ones(n, dtype=int)
        second_part = np.zeros(total_service - n, dtype=int)
        second_part = np.concatenate((second_part, first_part))
        np.random.shuffle(second_part)
        for i in zero_list:
            second_part = np.insert(second_part, i, 0)
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
    return np.random.uniform(vmin, vmax, size=(population, dimension))

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
    r1 = np.random.uniform(0, 1)
    r2 = np.random.uniform(0, 1)
    w = 1.2
    c1 = 2
    c2 = 2

    delta_velocity = c1 * r1 * (pbest - particle) + c2 * r2 * (gbest - particle)
    new_velocity = velocity + delta_velocity

    # Clamp velocities within bounds
    new_velocity = np.clip(new_velocity, vmin, vmax)

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
    pbest_function = np.array([func(p) for p in pbest_positions])
    gbest_index = np.argmin(pbest_function)
    gbest_position = pbest_positions[gbest_index]
    gbest_function = pbest_function[gbest_index]

    for i in range(iterations):
        for j in range(population):
            velocity[j] = update_velocity(particles[j], velocity[j], pbest_positions[j], gbest_position)
            particles[j] = update_dimensions(particles[j], velocity[j])
            current_function = func(particles[j])

            if current_function < pbest_function[j]:
                pbest_positions[j] = particles[j].copy()
                pbest_function[j] = current_function

            if current_function < gbest_function:
                gbest_position = particles[j].copy()
                gbest_function = current_function

    solution_time = time.time() - start_time
    return gbest_position, solution_time

best, soltime = particle_swarm(constraints, 550, 200, 275)
print("best:", constraints(best))
print("solution time:", soltime)



