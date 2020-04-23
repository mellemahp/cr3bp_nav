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
        cov (np.ndarray([[float]])): measurment covariance matrix

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
    def calc_msr(self):
        """Method that defines the equations that compute the
        measurement

        Note: users MUST overwrite this method in child classes

        """
        pass


    @abstractmethod
    def partials(self):
        """Computes the partial matrix with respect to the estimated
        state vector

        Note: users MUST overwrite this method in child classes

        """
        pass
