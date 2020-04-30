#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/
Description:Provides flexible measurement system for Advaned State estimation 
    class final project. Pulls from the OD_Suite software I wrote from 
    the 
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""

# standard library imports
from abc import ABC, abstractmethod
import numpy as np


class Msr(ABC):
    """Defines a base measurement class
    Args:
        state_vec (np.array([float])): states vector of the satellite
        stn (Station): station object associated with the measurement 
        time_tag (float): mod julian time at which this measurement was taken
        cov (np.ndarray([[float]])): measurement covariance matrix

    """
    def __init__(self, time_tag, msr, stn, cov):
        self.time = time_tag
        self.msr = msr
        self.stn = stn
        self.cov = cov

    @classmethod
    def from_stn(cls, time, state_vec, stn_vec, stn, cov):
        """ Initializes a measurement object from a station state vector
        """
        msr = cls.calc_msr(cls, state_vec, stn_vec)

        return cls(time, msr, stn, cov)


    def __repr__(self):
        string = """
        ===================================
        Measurement From Stn {} at time {}
        ===================================
        {}
        """.format(self.stn.stn_id, self.time, self.msr)


        return string


    def add_white_noise(self, sigma_vec):
        """ Adds gaussian noise to the measurement vector in place

        Args:
            sigma_vec (np.array): list of std deviations of size
                equal to the size of the measurement vector
        Raises:
            ValueError: The length of the sigma_vec is not equal to
                the size of the measurement vector

        """
        if len(sigma_vec) < len(self.msr):
            msg = "The length of the provided noise std deviation vector \
                {} does not match the size of the measurement vector \
                {}".format(len(sigma_vec),
                            len(self.msr))
            raise ValueError(msg)

        mean = [0, 0]
        cov_sigmas = np.diag([sigma**2 for sigma in sigma_vec])
        noise = np.random.multivariate_normal(mean, cov_sigmas, 1)
        self.msr = np.add(self.msr, noise)


    @abstractmethod
    def calc_msr(self, state_vec, station_vec):
        """Method that defines the equations that compute the
        measurement

        Note: users MUST overwrite this method in child classes

        """
        pass
    

class R3Msr(Msr):
    """Represents a RANGE and RANGE RATE measurement taken by a
    ground station
    Args:
        state_vec (list[adnumber]): list of parameters being estimated
            in the state vector. Should be dual numbers (using the
            adnumber package)
        stn_id (int): A unique int identifier for the station that took
            the measurement
        time_tag (float): mod julian time at which this measurement was
            taken
        cov (np.array([float])): measurment covariance matrix
    """
    def __init__(self, time_tag, msr, stn_id, cov):
        super(R3Msr, self).__init__(time_tag, msr, stn_id, cov)


    def calc_msr(self, state_vec, stn_state):
        """Calculates the instantaneous range and range rate measurement
        Args:
            state_vec (list[adnumber]): list of parameters being estimated
                in the state vector. Should be dual numbers (using the
                adnumber package)
            stn_vec (list[float || adnumber]): state vector of the
                station taking the measurement. Should be floats if
                the station is not being estimated. If the stn state
                is being estimated then adnumber with tagged names
                should be used instead
            time
        Return:
            list([1x2]): returns a 1 by 2 list of the range and
                range rate measurements
        """
        # if stn_state.any():
        stn_vec = stn_state
        rho = np.linalg.norm(np.subtract(state_vec[0:3], stn_vec[0:3]))
        rho_dot = np.dot(np.subtract(state_vec[0:3], stn_vec[0:3]),
                         np.subtract(state_vec[3:6], stn_vec[3:6]))/ rho

        return [rho, rho_dot]