# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:03:09 2019

@author: Jerry Xing
"""
import numpy as np


def grdescent(func, w0, stepsize, maxiter, tolerance=1e-2):
    # % function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
    # %
    # % INPUT:
    # % func function to minimize
    # % w0 = initial weight vector
    # % stepsize = initial gradient descent stepsize
    # % tolerance = if norm(gradient)<tolerance, it quits
    # %
    # % OUTPUTS:
    # %
    # % w = final weight vector
    # %
    w = w0
    ## << Insert your solution here

    w_trained = w0
    # w_trained = w0.reshape(-1, 1)
    los_bef, grad_bef = func(w_trained)
    while maxiter != 0 and np.linalg.norm(grad_bef) > tolerance:
        maxiter = maxiter - 1
        los_bef, grad_bef = func(w_trained)
        w_trained = w_trained - stepsize * grad_bef

    w = w_trained

    ## >>
    return w
