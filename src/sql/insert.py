#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/sql_schema.py
Description: Defines sqlite insert commands for adding rows to databases
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
class InsertMsr(object):
    """ Contianer object for insert commands on Measurement database """
    TRUTH_STATE = '''INSERT INTO truth_states(position, velocity, frame, timestamp, properties) 
                    VALUES(?,?,?,?,?)'''
    STN = '''INSERT INTO stations(
                    stn_id,
                    stn_name,
                    covariance,
                    latitude_deg, 
                    longitude_deg,
                    elevation_km, 
                    elevation_mask_deg,
                    measurement_types, 
                    mu)
            VALUES(?,?,?,?,?,?,?,?,?)'''
    MSR = '''INSERT INTO measurements(
                stn_id, 
                measurement, 
                timestamp,
                msr_type, 
                covariance)
            VALUES(?,?,?,?,?)'''

class InsertEst(object):
    """ Container object for insert commands on Estimates database """
    EST = '''INSERT INTO estimates(
                filter_name, 
                time, 
                state_est,
                covariance, 
                residual
            )
            VALUES(?,?,?,?,?)'''