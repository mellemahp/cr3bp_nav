#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/constants.py
Description: Defines various constants used throughout the project
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
#=== Begin Imports ===

# third party
import numpy as np

#=== End Imports ===

### Base unit conversions ###
DAYS_TO_SEC = 86400

### Non dimensional unit conversion ###
NON_DIM_DIST_TO_DIM = 384401 # Distance from earth to moon (km)
NON_DIM_TIME_TO_DIM = 4.3225 * DAYS_TO_SEC # period of Earth-moon system in sectonds

### Earth and Moon radii ###
R_E = 6378.1363 # radius of the earth (km)
R_E_ND = R_E * (1 / NON_DIM_DIST_TO_DIM) # radius of the earth in 
R_M = 1737.4
R_M_ND = R_M * (1 / NON_DIM_DIST_TO_DIM)

### Rotation rate of the earth ###
# W_E is defined for the ref epoch 3 Oct 1999, 23:11:9.1814
W_E = 7.2921158553e-5 # rotation rate of the earth (radians / sec)
W_E_ND = W_E *  NON_DIM_TIME_TO_DIM
THETA_0 = 0.0
EARTH_ANG_VEL = np.array([[0.0], [0.0], [W_E]])
EARTH_ANG_VEL_ND = np.array([[0.0], [0.0], [W_E_ND]])
# TODO: CHECK THIS
CR3BP_ANG_VEL = np.array([[0.0], [0.0], [2 * np.pi]])

# Values for calculating earth pole position
E_M_OFFSET_ANGLE = np.radians(28.7)
OMEGA_CR3BP = 1 / (2 * np.pi)


### Mu's for three body systems ###
MU_EARTH_MOON = 0.01215