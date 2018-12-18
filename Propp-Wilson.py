#!/usr/bin/env python
# coding: utf-8

# Perfect simulation of the Ferromagnetic Ising model.

import random
import numpy as np
from numpy.random import rand
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt


def InitialState(N):  # Generates an initial state with all values fixed to +1 and -1
    state = np.zeros((2, N, N))
    for i in range(N):
        for j in range(N):
            state[0][i][j] = 1
            state[1][i][j] = -1
    return state


def calcEnergyDiff(i, j, state):  # Calculate the energy at flipping the vertex at [i,j]
    m = state.shape[1] - 1
    if i == 0:
        top = 0
    else:
        top = state[i - 1, j]
    if i == m:
        bottom = 0
    else:
        bottom = state[i + 1, j]
    if j == 0:
        left = 0
    else:
        left = state[i, j - 1]
    if j == m:
        right = 0
    else:
        right = state[i, j + 1]
    energy = 2 * state[i][j] * sum([top, bottom, left, right])  # Energy calculated by given formula
    return energy


def updateState(t, state, U):
    B = 1 / T
    E = np.zeros(2)
    i = int(U[1][-t])  # Picks a random vertex, the same each time the chain runs from 0
    j = int(U[2][-t])
    for h in range(2):
        E[h] =  calcEnergyDiff(i, j, state[h])  # Find energy under randomly generated flip of each state space separately
        u = U[0][-t]
        if state[h][i][j] == 1:
            u = 1 - u
        if u < 0.5 * (1 - np.tanh(0.5 * B * E[h])):  # condition to accept change, random number is the same each time
            state[h][i][j] = -state[h][i][j]
            #n[h]=1
        else:
            state[h][i][j] = state[h][i][j]
            #n[h]=0

    if state[0][i][j] < state[1][i][j]:
        exit(1)
    return state  # returns both states


def runIsing(t, state, U):  # Runs chain from the designated starting time until time 0
    while t <= 0:
        #print("h=0",state[0])
        #print("h=1",state[1]
        state = updateState(t, state, U)
        t += 1
    return state


def genStartingTimes(j):  # Creates starting times, each one is double the previous
    M = [0] * j
    M[1] = 1
    for x in range(2, j):
        M[x] = 2 * M[x - 1]
    return M


def genRandomness(N, M):  # generate and store three sets of random numbers
    U = np.zeros((3, M))
    U[1] = np.random.randint(N, size=M)  # Random numbners i
    U[2] = np.random.randint(N, size=M)  # Random numbners j
    for i in range(0, M):
        U[0][i] = np.random.random()  # Random numbners U
    return U


def runProppWilson(N, j):
    M = genStartingTimes(j)
    U = genRandomness(N, 1)
    state = InitialState(N)
    m=1
    while not np.array_equal(state[0], state[1]):  # Condition for termination: both state spaces are the same
        U = np.append(U, genRandomness(N, M[m] - M[m - 1]), 1)  # Generates more random numbers when necessary
        magnetization = sum([sum(i) for i in state[0]])-sum([sum(i) for i in state[1]])
        print("magnetization= ",magnetization, "round= ", m)
        state = runIsing(-M[m], state, U)
        m += 1  # If states are not the same, goes to the next starting time
    return state[0]


def Graph(N, j):
    state = runProppWilson(N, j)
    S = state.shape[1]  # Takes the size of the matrix
    print("Plotting!")
    for i in range(S):
        for j in range(S):
            if state[i][j] == 1:  # Graphs a red + if the matrix entry is positive
                plt.scatter(j, S - 1 - i, c='r', marker=',', )#s=(150,))
            elif state[i][j] == -1:  # Graphs a blue minus is the matrix is negative
                plt.scatter(j, S - 1 - i, c='b', marker=',', )#s=(150,))
    plt.title("Sample, T=%d" % (T) )
    print("Done!")
    plt.show()


T = 2  # Temperature
Graph(50, 40) # Obtain a sample in a N x N grid



