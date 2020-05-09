#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/gmm.py
Description: Defines the gaussian mixture model for the project
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
# === Begin Imports ===

# third party
from numpy.linalg import LinAlgError, det
import random
from math import floor
import numpy as np
import scipy.stats
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
from scipy.linalg import eigh

# === End Imports ===


class GMM(object):
    """ 

    """

    def __init__(
        self,
        istate,
        msrs,
        apriori,
        force_model,
        jacobian,
        threshold_split=0.01,
        threshold_merge=0.1,
        max_components=20, 
        threshold_weight = 1e-7
    ):
        self.istate = istate
        self.estimates = [istate]
        self.msrs = msrs
        self.cov_list = [apriori]
        self.force_model = force_model
        self.jacobian = jacobian
        self.times = [0]
        self.len_state = len(istate)
        self.number_of_components = [1]
        self.components = [
            GMMComponentUKF(istate, 1.0, apriori, force_model, jacobian, 0.0)
        ]
        # gmm tuning params
        self.threshold_split = threshold_split
        self.threshold_merge = threshold_merge
        self.max_components = max_components
        self.threshold_weight = threshold_weight

    def run(self):
        count = 0
        for msr in self.msrs:
            components_to_add = []
            # propagation and splitting
            for component in self.components:
                component.run(msr)
                if component.failed == True:
                    if len(self.components) == 1: 
                        raise ValueError("The only remaining component failed")
                    self.components.remove(component)
                    continue
                
                if component.diff_entropy > self.threshold_split:
                    if len(self.components) < self.max_components:
                        components_to_add.extend(component.split())
                        self.components.remove(component)

            # add all new components
            self.components.extend(components_to_add)

            # reset weights
            w_n = 1.0 / len(self.components)
            total = sum([w_n * c.likelihood for c in self.components])
            if total == 0: 
                for component in self.components:
                    component.weight = w_n
            else: 
                for component in self.components:
                    component.weight = w_n * component.likelihood / total

            for component in self.components: 
                if component.weight < self.threshold_weight: 
                    if len(self.components) != 1:
                        self.components.remove(component)
                    else:
                        component.weight = 1.0
            
            # reset weights
            w_n = 1.0 / len(self.components)
            total = sum([w_n * c.likelihood for c in self.components])
            if total == 0: 
                for component in self.components:
                    component.weight = w_n
            elif len(self.components) == 1: 
                self.components[0].weight = 1
            else: 
                for component in self.components:
                    component.weight = w_n * component.likelihood / total
            
            # merging step. Randomly pair up and merge if divergence is below threshold
            components_to_add = []
            if len(self.components) > 4:
                for _ in range(floor(len(self.components) / 8)):
                    i, j = random.sample(range(0, len(self.components) - 1), 2)
                    a = self.components[int(i)]
                    b = self.components[int(j)]
                    dist = a.pseudo_mahala(b)
                    if dist < self.threshold_merge:
                        components_to_add.append(a.merge(b))
                        self.components.remove(a)
                        self.components.remove(b)

                # add all new components
                self.components.extend(components_to_add)

            # compute MMSE estimate
            x_mmse = sum([c.weight * c.mean for c in self.components])

            # compute error covariance
            weighted_tilde_sum = np.zeros((self.len_state, self.len_state))
            for c in self.components:
                p_tilde = c.cov + c.mean @ c.mean.T
                weighted_tilde_sum = weighted_tilde_sum + c.weight * p_tilde

            err_cov = weighted_tilde_sum - x_mmse @ x_mmse.T

            # update parameters
            self.cov_list.append(err_cov)
            self.estimates.append(x_mmse)
            self.times.append(msr.time)
            self.number_of_components.append(len(self.components))
            count = count + 1
            #print("Iter: {}".format(count), "Components: {}".format(len(self.components)))


class GMMComponentUKF(object):
    """Creates an Unscented Kalman Filter
    Args:
        istate (list[floats]): initials state vector. Will define the size of the state for
            all measurement processing

        force_model (filtering.ForceModel): force model to use for propagation and estimation
        alpha (float): scaling factor for sigma point selection (default=1e-3)
        beta (float): scaling factor for weighting of sigma points (default=2)
        kappa (float): scaling factor for sigam point selection (default=0)
    """

    def __init__(
        self,
        mean,
        weight,
        cov,
        force_model,
        jacobian,
        time,
        alpha=1e-3,
        beta=2,
        kappa=0,
        process_noise=None,
        alpha_split=0.6,
        beta_split=0.2,
        u_split=0.2,
    ):
        # main state values
        self.mean = mean
        self.len_state = len(mean)
        self.weight = weight
        self.cov = cov
        self.time = time
        self.diff_entropy = 0.0
        # force models
        self.force_model = force_model
        self.jacobian = jacobian
        # ukf tuning params
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha ** 2 * (kappa + self.len_state) - self.len_state
        self.gamma = np.sqrt(self.len_state + self.lam)
        # ukf weights
        w_m, w_c = self._get_weights()
        self.weights_sigs = w_m
        self.weights_covs = w_c
        # failure flag
        self.failed = False
        # process noise
        self.process_noise = process_noise
        # gmm tuning params
        self.alpha_split = alpha_split
        self.u_split = u_split
        self.beta_split = beta_split
        self.diff_ent_lin = 0.0
        self.likelihood = 1.0

    def _get_weights(self):
        """Finds the weights for the UKF
        Returns:
            (list[float], list[floats])
        """
        weights_sigs = [self.lam / (self.lam + self.len_state)]
        weights_cov = [weights_sigs[0] + (1 - self.alpha ** 2 + self.beta)]

        other_weights = 1 / (2 * (self.lam + self.len_state))

        for _ in range(1, 2 * self.len_state + 1):
            weights_sigs.append(other_weights)
            weights_cov.append(other_weights)

        return weights_sigs, weights_cov

    def prop_sigma_pts(self, t_f, sigma_points):
        """Maps the integration step to multiple processes
        Args:
            t_f (float): time to integrate to in seconds
            sigma_points (list[list[floats]]): list of 5 sigma points
        """
        outputs = []
        for sigma in sigma_points:
            outputs.append(self._integration_eq(self.time, t_f, sigma))

        return outputs

    def _integration_eq(self, t_0, t_f, X_0):
        """ Integrates a sigma point from on time to another
        Args:
            t_0 (float): start time in seconds (should be time of msr)
            t_f (float): time to integrate to
            X_0 (list[float]): intitial state to integrate
        """
        sol = solve_ivp(
            self.force_model, [t_0, t_f], X_0, method="LSODA", atol=1e-9, rtol=1e-6
        )
        X_f = sol.y[:, -1]

        return X_f

    def run(self, msr):
        """Runs the filter on the currently loaded measurement list
        """
        try:
            # generate sigma points
            sigma_points = self._find_sigma_pts(self.mean)

            # propagate sigma points
            sigma_points_prop = self.prop_sigma_pts(msr.time, sigma_points)

            # time update
            x_p = np.sum(
                [w * x for w, x in zip(self.weights_sigs, sigma_points_prop)], axis=0
            )

            P_i_m = np.sum(
                [
                    w * np.outer((x - x_p), (x - x_p))
                    for w, x in zip(self.weights_covs, sigma_points_prop)
                ],
                axis=0,
            )

            est_msrs = [self._msr_est(msr, x) for x in sigma_points_prop]

            y_hat_m = np.sum(
                [w * est for w, est in zip(self.weights_sigs, est_msrs)], axis=0
            )

            # measurement update
            p_yy_m = (
                np.sum(
                    [
                        w * np.outer((est - y_hat_m), (est - y_hat_m))
                        for w, est in zip(self.weights_covs, est_msrs)
                    ],
                    axis=0,
                )
                + msr.cov
            )
            p_xy_m = np.sum(
                [
                    w * np.outer((x - x_p), (est - y_hat_m))
                    for w, x, est in zip(self.weights_covs, sigma_points_prop, est_msrs)
                ],
                axis=0,
            )
            
            s_k = np.linalg.inv(p_yy_m)
            k_i = p_xy_m @ s_k

            resid = np.reshape(msr.msr, (len(msr.msr), 1)) - y_hat_m
            
            x_i_p = x_p + (k_i @ resid).T
            P_i_p = P_i_m - k_i @ (p_yy_m) @ k_i.T

            x_i_p = np.reshape(x_i_p, (1, self.len_state))[0]

            # store all relevant values
            self.cov = P_i_p
            self.mean = x_i_p
            self.diff_entropy = self._compute_diff_entropy(x_i_p, P_i_p, msr.time - self.time)
            self.time = msr.time
            
            # compute likelihood
            n_dist = scipy.stats.norm(msr.msr[0], s_k[0][0])
            self.likelihood = n_dist.pdf(y_hat_m[0][0])

        except ValueError:
            #print("A component borked")
            self.failed = True

    def _compute_diff_entropy(self, mean, cov, delta_t):
        """ """
        self.diff_ent_lin = self.diff_ent_lin + delta_t * np.trace(self.jacobian(mean)[3:,:3])
        return np.abs(
            self.diff_ent_lin - 2 * np.log(2 * np.pi * np.e) ** self.len_state * det(cov)
        )

    def _find_sigma_pts(self, mean):
        """Samples sigma points
        Args:
            mean (np.array [1 x n]): mean estimated state
            cov_sqrt (np.array [n x n]) sqrt of covariance matrix (scaled by function above)
        """
        cov_sqrt = self._get_mod_cov_sqrt(self.cov)
        mean_mat = np.array([mean for _ in range(self.len_state * 2 + 1)])
        mod_mat = np.block([[np.zeros((1, self.len_state))], [cov_sqrt], [-cov_sqrt]])
        sigs_mat = np.add(mean_mat, mod_mat)

        return sigs_mat

    def _get_mod_cov_sqrt(self, cov_mat):
        """Finds the scaled principal axis sqrt in
        Basically computes to principle axes of the uncertainty ellipsoid
        """
        S = scipy.linalg.sqrtm(cov_mat)

        if np.iscomplexobj(S):
            raise ValueError("Square root of covariance is complex: \n {}".format(S))

        return self.gamma * S

    def _msr_est(self, msr, state_prop):
        """ Computes the measurement residual and measurement partials
        Args:
            msr (filtering.MSR): measurement to use for computations
            state_prop (np.ndarray): nominal state vector propagated to the measurement time
        Returns:
            (np.ndarray [1 x len(MSR.msr)], np.ndarray [len(MSR.msr) x n])
        """
        # get estimated station position and estimated msr
        dummymsr = msr.__class__(msr.time, None, msr.stn, None)
        stn_state_est = msr.stn.state(msr.time)
        est_msr = dummymsr.calc_msr(state_prop, stn_state_est)

        return np.reshape(est_msr, (len(est_msr), 1))

    def merge(self, other):
        """ """
        diff = self.mean - other.mean
        weight_new = self.weight + other.weight
        mean_new = (
            1 / weight_new * (self.weight * self.mean + other.weight * other.mean)
        )

        cov_new = (
            1
            / weight_new
            * (
                self.weight * self.cov
                + other.weight * other.cov
                + self.weight * other.weight / weight_new * diff @ diff.T
            )
        )
        return GMMComponentUKF(
            mean_new, weight_new, cov_new, self.force_model, self.jacobian, self.time
        )

    def pseudo_mahala(self, other):
        """ """
        if self.weight != 0.0 and other.weight != 0.0:
            w = self.weight * other.weight / (self.weight + other.weight)
            diff = self.mean - other.mean
            return w * diff.T @ np.linalg.inv(self.cov) @ diff

        return 100

    def split(self):
        """ """
        _, eig_vecs = eigh(self.cov)
        a_l = np.array([eig_vecs[-1]])

        w_1 = self.alpha_split * self.weight
        w_2 = (1 - self.alpha_split) * self.weight

        mean_1 = self.mean - np.sqrt(w_2 / w_1) * self.u_split * a_l
        mean_2 = self.mean + np.sqrt(w_2 / w_1) * self.u_split * a_l
        P_1 = (
            w_2 / w_1 * self.cov
            + (self.beta_split - self.beta_split * self.u_split ** 2 - 1)
            * self.weight
            / w_1
            * a_l.T
            @ a_l
            + a_l.T @ a_l
        )
        P_2 = (
            w_1 / w_2 * self.cov
            + (
                self.beta_split * self.u_split ** 2
                - self.beta_split
                - self.u_split ** 2
            )
            * self.weight
            / w_2
            * a_l.T
            @ a_l
            + a_l.T @ a_l
        )

        return [
            GMMComponentUKF(
                mean_1[0], w_1, P_1, self.force_model, self.jacobian, self.time
            ),
            GMMComponentUKF(
                mean_2[0], w_2, P_2, self.force_model, self.jacobian, self.time
            ),
        ]
