#!/usr/bin/env python
"""Filters Module
Author: Hunter Mellema
Summary: Filters 

"""
# standard library imports
from abc import ABCMeta, abstractmethod

# local imports
from filtering.measurements import *
from filtering.propagator import *

# third party
import scipy as sp
from numba import jit
import numpy as np
import scipy.integrate as integrate
import scipy
import ad

# parallel stuff
from multiprocessing.pool import ThreadPool

class KalmanFilter(object):
    """ Defines a base class for all Kalman Filter type orbit determination filters
    Examples of child classes include:
        - Batch filter
        - Classic Kalman Filter
        - Extended Kalman Filter
    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation
    Note: All child classes are required to provide implementations of the following
        methods:
        - run()
        - measurement_update
    """
    __metaclass__ = ABCMeta

    def __init__(self, istate, msrs, apriori, force_model, process_noise=None):
        self.istate = istate
        self.prop_state_list = [istate]
        self.estimates = [istate]
        self.msrs = msrs
        self.apriori = apriori
        self.cov_list = [apriori]
        self.force_model = force_model
        self.residuals = [0]
        self.times = [0]
        self.len_state = len(istate)
        self.phis = []
        self.cov_ms = []
        self.process_noise = process_noise

    def __repr__(self):
        string="""
        {}
        ==================================
        Initial State:
        ++++++++++++++++++++++++++++++++++
        {}
        ++++++++++++++++++++++++++++++++++
        Msrs Processed: {}
        """.format(type(self), self.istate, len(self.estimates) - 1)

        return string

    @abstractmethod
    def run(self):
        """Defines how the filter will process measurements and update state
        estimates
        Note: Child classes MUST provide an implementation of this method
        """
        pass

    @abstractmethod
    def _measurement_update(self):
        """ Defines how measurements are used to update the state estimate
        Note: Child classes MUST provide an implementation of this method
        """
        pass


    def _compute_stm(self, time, phi=None):
        """Computes the STM by propagating it using an ode solver from
        the current time to the new time of the measurement
        Args:
           time (float): time in seconds past reference epoch to propagate STM to
        Returns:
           np.ndarray [n x n], np.ndarray [n x 1]
        """
        if phi is None:
            phi = np.identity(self.len_state)
            z_m = np.concatenate((self.prop_state_list[-1], phi.flatten()))
            t_0 = self.times[-1]
        else:
            phi = np.identity(self.len_state)
            z_m = np.concatenate((self.prop_state_list[0], phi.flatten()))
            t_0 = 0

        sol = solve_ivp(self._phi_ode,
                        [t_0, time],
                        z_m, method="LSODA",
                        atol=1e-8, rtol=1e-6
        )

        z_p = sol.y[:,-1]
        phi_p = np.reshape(z_p[self.len_state:],
                           (self.len_state,
                            self.len_state))

        prop_state = z_p[0:self.len_state]

        return phi_p, prop_state


    def _phi_ode(self, t, z):
        """Defines the STM ode equation. This is used only for STM propagation
        Args:
            t (float): time dummy variable for ode solver
            z (np.ndarray): a 1 x (n^2 + n) vector of the state vector and flattened STM matrix
        Returns:
           np.ndarray [1 x (n^2 + n)]
        """
        # prep states and phi_matrix
        state = z[0:self.len_state]

        # reshape to get
        phi = np.reshape(z[self.len_state:], (self.len_state,
                                              self.len_state))

        # find the accelerations and jacobian
        state_deriv, a_matrix = self._derivatives(t, state)

        # compute the derivative of the STM and repackage
        phid = np.matmul(a_matrix, phi)
        phid_flat = phid.flatten()
        z_out = np.concatenate((state_deriv, phid_flat))

        return z_out


    def _derivatives(self, t,  state):
        """ Computes the jacobian and state derivatives
        Args:
            state (np.ndarray): state vector to find derivatives of
        """
        ad_state = ad.adnumber(state)
        state_deriv = self.force_model.ode(t, ad_state)

        a_matrix = jacobian(state_deriv, ad_state)

        return state_deriv, a_matrix


    def _msr_resid(self, msr, state_prop):
        """ Computes the measurement residual and measurement partials
        Args:
            msr (filtering.MSR): measurement to use for computations
            state_prop (np.ndarray): nominal state vector propagated to the measurement time
        Returns:
            (np.ndarray [1 x len(MSR.msr)], np.ndarray [len(MSR.msr) x n])
        """
        # get estimated station position and estimated msr
        dummymsr = msr.__class__(msr.time, None, msr.stn, None)
        stn_state_est = msr.stn.get_state(msr.time)
        est_msr = dummymsr.calc_msr(state_prop, stn_state_est)

        # compute residuals and partials matrix
        y_i = np.subtract(msr.msr, est_msr).reshape(len(msr.msr), 1)
        h_tilde = msr.partials(ad.adnumber(state_prop), stn_state_est)

        return (y_i, h_tilde)


    def _calc_k_gain(self, cov_m, h_tilde, R_cov):
        """Calculates the Kalman Gain
        Args:
            cov_m: covariance matrix pre-measurement update
            h_tilde: measurement partials matrix
            R_cov: measurement covariance matrix
        Returns:
           np.ndarray [n x len(MSR.msr)]
        """
        B = np.matmul(cov_m, np.transpose(h_tilde))
        G = np.matmul(h_tilde, np.matmul(cov_m,
                                         np.transpose(h_tilde)))
        T = np.linalg.inv(np.add(G, R_cov))

        return np.matmul(B, T)

    def _compute_SNC(self, next_time):
        """Computes the SNC covariance matrix update """
        dt = self.times[-1] - next_time
        Q_k = np.zeros((self.len_state, self.len_state))
        Q_k[0,0], Q_k[1,1], Q_k[2,2] = [1 / 3 * dt**3] * 3
        Q_k[3,3], Q_k[4,4], Q_k[5,5] = [1 / 2 * dt**2] * 3
        Q_k[0,3], Q_k[1,4], Q_k[2,5], Q_k[3,0], Q_k[4,1], Q_k[5,2] = [dt] * 6

        Q_k = self.process_noise**2 * Q_k

        return Q_k


class CKFilter(KalmanFilter):
    """ Initializes a Kalman Filtering object that generate a state estimate from
    a list of measurements
    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        pert_vec (list[float]): intitial perturbation vector guess
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation
    """
    def __init__(self, istate, msrs, apriori, force_model, pert_vec, process_noise=None):
        super().__init__(istate, msrs, apriori, force_model, process_noise)

        self.pert_vec = [pert_vec]
        self.innovation = [0]

    def run(self):
        """Runs the filter on the currently loaded measurement list
        """
        for msr in self.msrs:
            # find state transition matrix
            phi_p, state_prop = self._compute_stm(msr.time)

            # use stm to propagate perturbation and covariance
            pert_m = np.matmul(phi_p, self.pert_vec[-1])
            cov_m = np.matmul(phi_p, np.matmul(self.cov_list[-1],
                                               np.transpose(phi_p)))

            # add process noise if there is any
            if self.process_noise:
                process_noise = self._compute_SNC(msr.time)
                cov_m = np.add(cov_m, process_noise)

            # compute observation deviation, obs_state matrix
            y_i, h_tilde = self._msr_resid(msr, state_prop)

            # calculate kalman gain
            k_gain = self._calc_k_gain(cov_m, h_tilde, msr.cov)

            # measurement update
            cov_p, pert_p = self._measurement_update(y_i,
                                                     h_tilde,
                                                     pert_m,
                                                     k_gain,
                                                     cov_m,
                                                     msr.cov
            )

            # update the state lists
            self.residuals.append(y_i)
            self.prop_state_list.append(state_prop)
            self.estimates.append(np.add(state_prop, np.transpose(pert_p))[0])
            self.cov_list.append(cov_p)
            self.cov_ms.append(cov_m)
            self.pert_vec.append(pert_p)
            self.times.append(msr.time)
            self.phis.append(phi_p)



    def _measurement_update(self, y_i, h_tilde, pert_m, k_gain, cov_m, msr_cov):
        """ Performs the measurement update step of the CKF using the
        Joseph covariance update
        Args:
            y_i (np.ndarray): measurement residuals matrix
            h_tilde (np.ndarray): measurement partials matrix
            pert_m (np.ndarray): perturbation vector propagated to the current time
                pre measurement update
            k_gain (np.ndarray): kalman gain matrix
            cov_m (np.ndarray [n x n]): covariance matrix propagated to current time pre
                measurement update
        Returns:
            (np.ndarray [n x n], np.ndarray [n x 1])
        """
        innovation = np.subtract(y_i, np.matmul(h_tilde, pert_m))
        self.innovation.append(innovation)
        pert_p = np.add(pert_m, np.matmul(k_gain, innovation))
        L = np.subtract(np.identity(self.len_state),
                        np.matmul(k_gain, h_tilde))
        Z = np.matmul(k_gain, np.matmul(msr_cov, np.transpose(k_gain)))
        Q = np.matmul(L, np.matmul(cov_m, np.transpose(L)))
        cov_p = np.add(Q, Z)

        return (cov_p, pert_p)

    def smooth(self):
        """ Applies smoothing to the computed estimates
        """
        self.S = [self.cov_list[-2] @ self.phis[-1].T @ np.linalg.inv(self.cov_ms[-1])]
        self.smoothed_perts = [
            self.pert_vec[-2] + self.S[-1] @ (self.pert_vec[-1] - self.phis[-1] @ self.pert_vec[-2])
        ]
        self.smoothed_states = [np.add(self.prop_state_list[-1], self.smoothed_perts[-1].T)[0]]
        self.smoothed_cov = [self.cov_list[-1]]

        for idx, pert in enumerate(self.pert_vec[:-1][::-1]):
            try:
                self.S.append(
                    self.cov_list[-idx-2] @ self.phis[-idx-1].T @ np.linalg.inv(self.cov_ms[-idx-1])
                )
                self.smoothed_perts.append(
                     self.pert_vec[-idx-2] + self.S[-1] @ (self.pert_vec[-idx] -
                                                       self.phis[-idx-1] @ self.pert_vec[-idx-2])
                )
                self.smoothed_states.append(
                    np.add(self.prop_state_list[-idx-1], self.smoothed_perts[-1].T)[0]
                )
                self.smoothed_cov.append(
                    np.add(self.cov_list[-idx-2],
                           self.S[-1] @ np.subtract(self.smoothed_cov[-1],
                                                    self.cov_ms[-idx-1]) @ self.S[-1].T)
                )

            except IndexError:
                break




class EKFilter(KalmanFilter):
    """ Initializes an Extended Kalman Filtering object that generate a state estimate from
    a list of measurements
    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing
        msrs (list[filtering.MSR]): list of measurements to process
        apriori (np.ndarray): apriori covariance matrix. Must have size n x n
        force_model (filtering.ForceModel): force model to use for propagation and estimation
    """
    def __init__(self, istate, msrs, apriori, force_model):
        super().__init__(istate, msrs, apriori, force_model)

    def run(self):
        """Runs the filter on the currently loaded measurement list
        """
        for msr in self.msrs:
            # find state transition matrix
            phi_p, state_prop = self._compute_stm(msr.time)

            # use stm to propagate perturbation and covariance
            cov_m = np.matmul(phi_p, np.matmul(self.cov_list[-1],
                                               np.transpose(phi_p)))

            # add process noise if there is any
            if self.process_noise:
                cov_m = np.add(cov_m, self._compute_SNC(msr.time))

            # compute observation deviation, obs_state matrix
            y_i, h_tilde = self._msr_resid(msr, state_prop)

            # calculate kalman gain
            k_gain = self._calc_k_gain(cov_m, h_tilde, msr.cov)

            # measurement update
            cov_p, state_est = self._measurement_update(y_i,
                                                        h_tilde,
                                                        k_gain,
                                                        cov_m,
                                                        state_prop)

            # update the state lists
            self.residuals.append(y_i)
            self.prop_state_list.append(state_est)
            self.estimates.append(state_est)
            self.cov_list.append(cov_p)
            self.times.append(msr.time)


    def _measurement_update(self, y_i, h_tilde, k_gain, cov_m, state_prop):
        """ Performs the measurement update step of the EKF
        Args:
            y_i (np.ndarray): measurement residuals matrix
            h_tilde (np.ndarray): measurement partials matrix
            pert_m (np.ndarray): perturbation vector propagated to the current time
                pre measurement update
            k_gain (np.ndarray): kalman gain matrix
            cov_m (np.ndarray [n x n]): covariance matrix propagated to current time pre
                measurement update
        Returns:
            np.ndarray [n x n], np.ndarray [1 x n]
        """
        x_update = np.matmul(k_gain, y_i)

        L = np.subtract(np.identity(len(self.istate)),
                        np.matmul(k_gain, h_tilde))

        cov_p = np.matmul(L, cov_m)

        state_est = np.add(state_prop, np.transpose(x_update))[0]

        return (cov_p, state_est)