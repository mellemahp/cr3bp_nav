#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/cr3bp.py
Description: Defines the base three body problem system
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
#=== Begin Imports ===

# third party
import numpy as np
from numpy.linalg import norm

# local imports
from .dynamics_partials import cr3bp_jacobian

#=== End Imports ===
        
class CR3BPSystem(object):
    """Sets up a three body system with the specified mu

    Args: 
        mu (float): three body system constant

    """
    def __init__(self, mu): 
        self.mu = mu

    def derivative(self, _t, state):
        """Finds the derivative of the system [dr, ddr]

        Args: 
            state (np.array): state vector of satellite in non-dimensional
                circularly restricted three body coordinates

        Returns: 
            np.array

        Equations From "Space Manifold Design"
        Note: all of the values here are _nondimensional_ to get dimensional
        values please use the `conversion` method 

        """
        x, y, z, dx, dy, dz = state
        r_13_inv = 1 / norm(self.r_1(self.mu, state))**3
        r_23_inv = 1 / norm(self.r_2(self.mu, state))**3

        ddx = x + 2 * dy - (1 - self.mu) * (x - self.mu) * r_13_inv \
                - (self.mu * (x + 1 - self.mu)) * r_23_inv
        ddy = y - 2 * dx - (1 - self.mu) * y * r_13_inv \
                - self.mu * y * r_23_inv
        ddz = -z * (1 - self.mu) * r_13_inv - self.mu * z * r_23_inv

        return np.array([dx, dy, dz, ddx, ddy, ddz])

    def jacobian(self, state): 
        return cr3bp_jacobian(self.mu, state)

    @staticmethod            
    def r_1(mu, state): 
        """Position Vector from the primary body (i.e. Earth) to the satellite
        
        Args: 
            state (np.array): state vector of satellite in non-dimensional
                circularly restricted three body coordinates

        Returns: 
            np.array

        """
        return [(state[0] - mu), state[1], state[2]]

    @staticmethod
    def r_2(mu, state):
        """Position Vector from secondary body (i.e. Moon) to the satellite
        
        Args: 
            state (np.array): state vector of satellite in non-dimensional
                circularly restricted three body coordinates

        Returns: 
            np.array

        """ 
        return [(state[0] + 1 - mu), state[1], state[2]]

    
    