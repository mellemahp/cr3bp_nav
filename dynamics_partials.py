#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/dynamics_partials.py
Description: Defines the partials and Jacobian for the dynamics of the CR3BP
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
#=== Begin Imports ===

# third party
import numpy as np
from numpy.linalg import norm

# local imports
from cr3bp import CR3BPSystem

#=== End Imports ===

#=========================
# X Velocity Partials
#=========================
dx_d_x = 0
dx_d_y = 0
dx_d_z = 0
dx_d_dx = 1
dx_d_dy = 0
dx_d_dz = 0

#=========================
# Y Velocity Partials
#=========================
dy_d_x = 0
dy_d_y = 0
dy_d_z = 0
dy_d_dx = 0
dy_d_dy = 1
dy_d_dz = 0

#=========================
# Z Velocity Partials
#=========================
dz_d_x = 0
dz_d_y = 0
dz_d_z = 0
dz_d_dx = 0
dz_d_dy = 0
dz_d_dz = 1


#=========================
# X Acceleration Partials
#=========================
def xddot_d_x(mu, state, r_13_inv, r_23_inv, r_15_inv, r_25_inv):
    """ Partial of x acceleration with respect to x

    Args:
        mu (float): three body constant
        state (np.array): 6 dimensional state vector of
            (x, y, z, dx, dy, dz)
        r_13_inv (float):  1 / norm(r_1)^(3) where r_1 is the vector from the
            primary (i.e. Earth) to the satellite
        r_23_inv (float):  1 / norm(r_2)^(3)  where r_2 is the vector from the
            secondary (i.e. Moon) to the satellite
        r_15_inv (float):  1 / norm(r_1)^(5)
        r_25_inv (float):  1 / norm(r_2)^(5)

    Returns:
        float

    """
    x, y, z = state[:3]

    ans = 3 * mu * (-mu + x + 1) * (-mu + x + 1) * r_25_inv \
            + 3 * (1 - mu) * (x - mu)**2 * r_15_inv \
            - (1 - mu) * r_13_inv \
            - mu * r_23_inv \
            + 1

    return ans


def xddot_d_y(mu, state, r_15_inv, r_25_inv):
    """ Partial of x acceleration with respect to y

    Args:
        mu (float): three body constant
        state (np.array): 6 dimensional state vector of
            (x, y, z, dx, dy, dz)
        r_15_inv (float):  1 / norm(r_1)^(5) where r_1 is the vector from the
            primary (i.e. Earth) to the satellite
        r_25_inv (float):  1 / norm(r_2)^(5) where r_2 is the vector from the
            secondary (i.e. Moon) to the satellite

    Returns:
        float
    """
    x, y, z = state[:3]

    ans = 3 * mu * y * (-mu + x + 1) * r_25_inv \
            + 3 * (1 - mu) * y * (x - mu) * r_15_inv

    return ans


def xddot_d_z(mu, state, r_15_inv, r_25_inv):
    """ Partial of x acceleration with respect to z

    Args:
        mu (float): three body constant
        state (np.array): 6 dimensional state vector of
            (x, y, z, dx, dy, dz)
        r_15_inv (float):  1 / norm(r_1)^(5) where r_1 is the vector from the
            primary (i.e. Earth) to the satellite
        r_25_inv (float):  1 / norm(r_2)^(5) where r_2 is the vector from the
            secondary (i.e. Moon) to the satellite

    Returns:
        float
    """
    x, y, z = state[:3]

    ans = 3 * mu * z * (-mu + x + 1) * r_25_inv \
            + 3 * (1 - mu) * z * (x - mu) * r_15_inv

    return ans

xddot_d_dx = 0
xddot_d_dy = 2
xddot_d_dz = 0

#=========================
# Y Acceleration Partials
#=========================

yddot_d_x = xddot_d_y

def yddot_d_y(mu, state, r_13_inv, r_23_inv, r_15_inv, r_25_inv):
    """ Partial of x acceleration with respect to y

    Args:
        mu (float): three body constant
        state (np.array): 6 dimensional state vector of
            (x, y, z, dx, dy, dz)
        r_13_inv (float):  1 / norm(r_1)^(3) where r_1 is the vector from the
            primary (i.e. Earth) to the satellite
        r_23_inv (float):  1 / norm(r_2)^(3) where r_2 is the vector from the
            secondary (i.e. Moon) to the satellite
        r_15_inv (float):  1 / norm(r_1)^(5)
        r_25_inv (float):  1 / norm(r_2)^(5)

    Returns:
        float
    """
    x, y, z = state[:3]

    ans = 3 * (1 - mu) * y**2 * r_15_inv \
            + 3 * mu * y**2 * r_25_inv \
            - (1 - mu) * r_13_inv \
            - mu * r_23_inv \
            + 1

    return ans


def yddot_d_z(mu, state, r_15_inv, r_25_inv):
    """ Partial of x acceleration with respect to z

    Args:
        mu (float): three body constant
        state (np.array): 6 dimensional state vector of
            (x, y, z, dx, dy, dz)
        r_15_inv (float):  1 / norm(r_1)^(5) where r_1 is the vector from the
            primary (i.e. Earth) to the satellite
        r_25_inv (float):  1 / norm(r_2)^(5) where r_2 is the vector from the
            secondary (i.e. Moon) to the satellite

    Returns:
        float
    """
    x, y, z = state[:3]

    ans = 3 * mu * y * z * r_25_inv \
            + 3 * (1 - mu) * y * z * r_15_inv

    return ans


yddot_d_dx = -2
yddot_d_dy = 0
yddot_d_dz = 0

#=========================
# Z Acceleration Partials
#=========================
zddot_d_x = xddot_d_z
zddot_d_y = yddot_d_z

def zddot_d_z(mu, state, r_13_inv, r_23_inv, r_15_inv, r_25_inv):
    """ Partial of x acceleration with respect to z

    Args:
        mu (float): three body constant
        state (np.array): 6 dimensional state vector of
            (x, y, z, dx, dy, dz)
        r_13_inv (float):  1 / norm(r_1)^(3) where r_1 is the vector from the
            primary (i.e. Earth) to the satellite
        r_23_inv (float):  1 / norm(r_2)^(3) where r_2 is the vector from the
            secondary (i.e. Moon) to the satellite
        r_15_inv (float):  1 / norm(r_1)^(5)
        r_25_inv (float):  1 / norm(r_2)^(5)

    Returns:
        float
    """
    x, y, z = state[:3]

    ans = -3 * (mu - 1) * z**2 * r_15_inv \
            + 3 * mu * z**2 * r_25_inv \
            + (mu - 1) * r_13_inv \
            - mu * r_23_inv

    return ans


zddot_d_dx = 0
zddot_d_dy = 0
zddot_d_dz = 0


#=========================
# Jacobian
#=========================
def cr3bp_jacobian(mu, state):
    """Returns the Jacobian Matrix for the Circularly Restricted 3 Body Problem

    Args:
        mu (float): three body constant
        state (np.array): 6 dimensional state vector of
            (x, y, z, dx, dy, dz)

    Returns:
        np.array: 6x6 Matrix

    """
    r_1_norm = norm(CR3BPSystem.r_1(mu, state))
    r_2_norm = norm(CR3BPSystem.r_2(mu, state))
    r_13_inv = 1 / r_1_norm**3
    r_23_inv = 1 / r_2_norm**3
    r_15_inv = r_13_inv * 1 / r_1_norm**2
    r_25_inv = r_23_inv * 1 / r_2_norm**2

    return np.array([
        [dx_d_x, dx_d_y, dx_d_z, dx_d_dx, dx_d_dy, dx_d_dz],
        [dy_d_x, dy_d_y, dy_d_z, dy_d_dx, dy_d_dy, dy_d_dz],
        [dz_d_x, dz_d_y, dz_d_z, dz_d_dx, dz_d_dy, dz_d_dz],
        [
            xddot_d_x(mu, state, r_13_inv, r_23_inv, r_15_inv, r_25_inv),
            xddot_d_y(mu, state, r_15_inv, r_25_inv),
            xddot_d_z(mu, state, r_15_inv, r_25_inv),
            xddot_d_dx,
            xddot_d_dy,
            xddot_d_dz
        ],
        [
            yddot_d_x(mu, state, r_15_inv, r_25_inv),
            yddot_d_y(mu, state, r_13_inv, r_23_inv, r_15_inv, r_25_inv),
            yddot_d_z(mu, state, r_15_inv, r_25_inv),
            yddot_d_dx,
            yddot_d_dy,
            yddot_d_dz
        ],
        [
            zddot_d_x(mu, state, r_15_inv, r_25_inv),
            zddot_d_y(mu, state, r_15_inv, r_25_inv),
            zddot_d_z(mu, state, r_13_inv, r_23_inv, r_15_inv, r_25_inv),
            zddot_d_dx,
            zddot_d_dy,
            zddot_d_dz
        ]
    ])


#=========================
# Test Functions
#=========================


# Runs test of the dynamics
if __name__ == "__main__":
    from numdifftools import Jacobian
    from constants import MU_EARTH_MOON
    from functools import partial
    from random import randint

    cr3bp_system = CR3BPSystem(MU_EARTH_MOON)
    TOL = 1e-6
    NUM_TESTS = 1000

    for _ in range(NUM_TESTS):
        test_state = [randint(0, 100) * 0.33 for _ in range(6)]
        jacobian_true = Jacobian(cr3bp_system.derivative)(test_state)
        jacobian_est = cr3bp_jacobian(MU_EARTH_MOON, test_state)

        for i in range(6):
            for j in range(6):
                if abs(jacobian_est[i][j] - jacobian_true[i][j]) > TOL:
                    with np.printoptions(precision=3, suppress=True):
                        print("ELEMENT [{}][{}] Does not match".format(i,j))
                        print("EST: {}".format(jacobian_est[i][j]))
                        print("TRUE: {}".format(jacobian_true[i][j]))
                        print("===== True ======= \n", jacobian_true, "\n ===============")
                        print("===== Est ======== \n", jacobian_est, "\n ===============")
                        raise AssertionError("True and Estimated Jacobians do not match")

    print("{} Tests Passed".format(NUM_TESTS))

    exit()