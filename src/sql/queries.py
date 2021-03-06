#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/sql_schema.py
Description: Defines sqlite queries for data retrieval
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
class GetMsr(object):
    """ Container object for getting data from the Measurement table """
    LAST_TRUTH_STATE_TIMES = "SELECT timestamp FROM truth_states ORDER BY timestamp DESC LIMIT 1;"
    ALL_TRUTH_STATES = "SELECT position, velocity, timestamp FROM truth_states ORDER BY timestamp ASC;"
    ALL_TRUTH_STATE_AND_TIMES = "SELECT position, velocity, timestamp FROM truth_states ORDER BY timestamp ASC;"
    ALL_MSRS = "SELECT stn_id, measurement, timestamp, msr_type, covariance FROM measurements ORDER BY timestamp ASC;"

class GetEst(object): 
    FILTER_ESTS = "SELECT time, state_est, covariance, residual FROM estimates WHERE filter_name IS ? ORDER BY time ASC;"