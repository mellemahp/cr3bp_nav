#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/sql_schema.py
Description: Defines sqlite schema for use in data storage
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
class CreateMsr(object):
    """ Container object for Commands to Create tables in the Measurement DB """
    STN_TABLE = '''CREATE TABLE IF NOT EXISTS stations(
                    id integer PRIMARY KEY,
                    stn_id integer NOT NULL, 
                    stn_name string NOT NULL,
                    covariance string NOT NULL, 
                    latitude_deg numeric NOT NULL, 
                    longitude_deg numeric NOT NULL, 
                    elevation_km numeric NOT NULL, 
                    elevation_mask_deg numeric NOT NULL,
                    measurement_types string,
                    mu numeric NOT NULL
                );'''

    MSR_TABLE = '''CREATE TABLE IF NOT EXISTS measurements(
                    id integer PRIMARY KEY,
                    stn_id integer NOT NULL, 
                    measurement string NOT NULL, 
                    timestamp numeric NOT NULL,
                    msr_type string NOT NULL, 
                    covariance string NOT NULL
                );'''
                        
    TRUTH_STATES_TABLE = '''CREATE TABLE IF NOT EXISTS truth_states(
                            id integer PRIMARY KEY,
                            position string NOT NULL, 
                            velocity string NOT NULL, 
                            frame string NOT NULL, 
                            timestamp numeric NOT NULL, 
                            properties string
                        );'''       

class CreateEst(object):
    """ Container object for Commands to Create tables in the Estimates DB """
    EST_TABLE = '''CREATE TABLE IF NOT EXISTS estimates(
                    id integer PRIMARY KEY,
                    filter_name integer NOT NULL, 
                    time numeric NOT NULL, 
                    state_est string NOT NULL,
                    covariance string NOT NULL, 
                    residual string NOT NULL
                );'''