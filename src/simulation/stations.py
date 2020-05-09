#!/urs/bin/env python
# -*- coding: utf-8 -*-
"""/stations.py
Author: Hunter Mellema
Summary: Defines measuring stations for simulation for CR3BP navigation simulations
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
# === Begin Imports ===
# third party
import numpy as np
from numpy.linalg import norm

# local imports
from .constants import (
    THETA_0,
    R_E,
    W_E_ND,
    R_E_ND,
    EARTH_ANG_VEL_ND,
    NON_DIM_DIST_TO_DIM,
    MU_EARTH_MOON,
    E_M_OFFSET_ANGLE, 
    OMEGA_CR3BP, 
    CR3BP_ANG_VEL
)
from .measurements import R3Msr, Range

# std library
import json

# === End Imports ===

# Measurement Generation Functions
class CR3BPEarthStn(object):
    """ Represents a ground station at some point on the earth's surface that is taking measurements
    of a spacecraft.
    Args:
        stn_id (str, int): a unique string or integer id for the given station
        longitude (float): longitude of the station position in degrees
        latitude (float): latitude of the station in degrees
        el_mask (float): elevation below which the station will not take measurements (in degrees)
        covariance (np.ndarray): covariance for measurements take by this station. Will be added to
            every measurement taken by this station

    """

    def __init__(
        self,
        stn_name,
        stn_id,
        pos_ecef_nd,
        el_mask,
        cov,
        msrs,
        mu,
        lat=None,
        longitude=None,
        el=None,
    ):
        self.stn_name = stn_name
        self.stn_id = stn_id
        self.pos = pos_ecef_nd
        self.el_mask = el_mask
        self.cov = cov
        self.allowed_msrs = msrs
        self.lat = lat
        self.long = longitude
        self.el = el
        self.mu = mu

    @classmethod
    def from_db_object(cls, station_data):
        """ Constructs a stn from a row in the database
        
        Args: 
            station_data (dict): dicitonary representing a row in the stations table

        Return: 
            cls

        """
        pos_nd = lat_long_to_ecef(
            station_data["latitude_deg"],
            station_data["longitude_deg"],
            station_data["elevation_km"],
        ) * (1 / NON_DIM_DIST_TO_DIM)

        return cls(
            station_data["stn_name"],
            station_data["stn_id"],
            pos_nd,
            station_data["elevation_mask_deg"],
            json.loads(station_data["covariance"]),
            json.loads(station_data["measurement_types"]),
            station_data["mu"],
            station_data["latitude_deg"],
            station_data["longitude_deg"],
            station_data["elevation_km"],
        )

    def __repr__(self):
        string = """
        ==================================================
        ++++++++++++++++++++++++++++++++++++++++++++++++++
        STN: {}     | STN ID: {}
        ++++++++++++++++++++++++++++++++++++++++++++++++++
        Position: 
            {} (Nondimensional CR3BP)
        Location (lat, long, el): 
            {} deg N, {} deg E, {} Km,
        Elevation Mask: {} deg
        Measurement Covariance: {}
        ==================================================
        """.format(
            self.stn_name,
            self.stn_id,
            self.pos,
            self.lat,
            self.long,
            self.el,
            self.el_mask,
            self.cov,
        )

        return string

    def gen_msr(self, sc_state, time, msr_type="R3Msr"):
        """Generates a measurement if the spacecraft is visible

        Args:
            sc_state (np.ndarray): spacecraft state vector, pos, vel must be first 6 terms
            time (float): time since reference epoch
            msr_type (str): name of measurement to use (default = R3 (range and range rate))

        Returns:
            tuple(bool, float, np.ndarray)

        """
        valid, el, stn_state = self._check_elevation(sc_state, time)
        if valid:
            return globals()[msr_type].from_stn(
                time, sc_state, stn_state, self.stn_id, self.cov
            )
        else:
            return None

    def _check_elevation(self, sc_state, time):
        """Checks to see if the spacecraft is visible

        Args:
            sc_state (np.ndarray): spacecraft state vector, pos, vel must be first 6 terms
            time (float): time since reference epoch

        Returns:
            tuple(bool, float, np.ndarray)

        """
        stn_state_ns = self.state(time, include_shift=False)
        stn_state_shift = stn_state_ns + np.array([self.mu, 0.0, 0.0, 0.0, 0.0, 0.0])
        line_o_sight = sc_state[0:3] - stn_state_shift[0:3]
        num = np.dot(stn_state_ns[0:3], line_o_sight)
        denom = np.linalg.norm(stn_state_ns[0:3]) * np.linalg.norm(line_o_sight)
        zenel = np.arccos(num / denom)
        el = np.pi / 2 - zenel

        if el > np.deg2rad(self.el_mask):
            flag = True
        else:
            flag = False

        return (flag, el, stn_state_shift)

    def state(self, time_nd, include_shift=True):
        """Finds the station state in ECI at a given nondimensional CR3BP time

        Args:
            time_nd (float): non-dimensional CR3BP time past reference epoch

        Returns:
            np.ndarray([1x6])

        """
        rot_mat_ecef_eci = ecef_to_eci_nd(time_nd)
        rot_mat_eci_cr3bp = eci_to_cr3bp(time_nd)
        pos_eci = rot_mat_ecef_eci @ self.pos
        pos_cr3bp = rot_mat_eci_cr3bp @ pos_eci
        
        if include_shift:
            pos_cr3bp = pos_cr3bp + np.array([self.mu, 0.0, 0.0])

        vel_ecef = np.cross(EARTH_ANG_VEL_ND.T, self.pos.T).T
        vel_eci = rot_mat_ecef_eci @ vel_ecef
        vel_cr3bp_no_cr3bp_and_vel = rot_mat_eci_cr3bp @ vel_eci
        coriolis = np.cross(CR3BP_ANG_VEL.T, pos_cr3bp.T).T
        vel_cr3bp = vel_cr3bp_no_cr3bp_and_vel + coriolis

        state = np.concatenate((pos_cr3bp.T[0], vel_cr3bp.T[0]))

        return state


def lat_long_to_ecef(latitude, longitude, elevation):
    """Converts from latituted-longitude to cartesian position in ECEF
    Note: Assumes location is on the surface of the earth and uses a spherical
        earth model

    Args:
        latitude (float): latitude to convert (in degrees)
        longitude (float): longitude to convert (in degrees)
        elevation (float): elevation above spherical model (in km)

    Returns:
       np.ndarray([3x3])

    """
    phi = np.deg2rad(latitude)
    lam = np.deg2rad(longitude)

    pos_ecef = (R_E + elevation) * np.array(
        [[np.cos(phi) * np.cos(lam)], [np.cos(phi) * np.sin(lam)], [np.sin(phi)]]
    )

    return pos_ecef


def ecef_to_eci_nd(time_nd, theta_0=THETA_0):
    """Calculates rotation matrix to ECI at a given time
    Args:
       time_nd (float): nondimensional time since reference epoch
       theta_0 (float): initial rotation of the earth at the reference epoch
            [default=filtering.THETA_0]
    Returns:
       np.ndarray([3x3])
    """
    alpha = theta_0 + time_nd * W_E_ND
    rot_mat = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )

    return rot_mat


def eci_to_cr3bp(time_nd):
    """Calculates rotation matrix from ECI to CR3BP at a given nondimensional time

    Args:
       time_nd (float): nondimensional time since reference epoch

    Returns:
       np.ndarray([3x3])

    """
    axis_north_pole = np.array([
        np.sin(-E_M_OFFSET_ANGLE) * np.cos(time_nd * OMEGA_CR3BP), 
        np.sin(-E_M_OFFSET_ANGLE) * np.sin(time_nd * OMEGA_CR3BP), 
        np.cos(-E_M_OFFSET_ANGLE)
    ])
    z_axis = np.array([0.0, 0.0, 0.1])
    rot = rotate_plane(z_axis, axis_north_pole)

    return rot


def unitcross(a, b): 
    c = np.cross(a, b)
    return c / norm(c)

def costheta(a, b): 
    return np.dot(a, b) / (norm(a) * norm(b))

def rotate_plane(M, N): 
    (x, y, z) = unitcross(M, N)
    c = costheta(M, N)
    s = np.sqrt(1-c*c)
    C = 1-c
    rmat = np.array(
        [
            [ x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [ y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [ z * x * C - y * s, z * y * C + x * s, z * z * C + c]
        ]
    )

    return rmat