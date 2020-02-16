import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, sinh, pi
import scipy as sp


def __gauss(LAMBDA, MU, SIGMA):
    a = 1/(2*SIGMA**2)
    def f(x):
        return LAMBDA * np.exp(-(x - MU)**2 / a)
    return f

def __johnson (GAMMA, SIGMA, LAMBDA, XI):
    a = SIGMA/(LAMBDA * sqrt(2*pi))
    l = 1/LAMBDA
    def f(x):
        z = (x-XI)*l
        return a/np.sqrt(z**2+1)*np.exp(-0.5*(GAMMA+SIGMA*np.arcsinh(z))**2)
    return f