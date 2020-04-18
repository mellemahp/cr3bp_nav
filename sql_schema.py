#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/sql_schema.py
Description: Defines sqlite schema for use in data storage
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
# === Begin Imports ===
MSR_CREATE_MSR_TABLE = '''CREATE TABLE IF NOT EXISTS measurements(
                            id integer PRIMARY KEY,
                            stn_id integer NOT NULL, 
                            measurement string NOT NULL, 
                            timestamp numeric NOT NULL,
                            msr_type  string NOT NULL, 
                            covariance string NOT NULL
                        );'''
                        
MSR_CREATE_TRUTH_TABLE = '''CREATE TABLE IF NOT EXISTS truth_states(
                            id integer PRIMARY KEY,
                            position string NOT NULL, 
                            velocity string NOT NULL, 
                            frame string NOT NULL, 
                            timestamp numeric NOT NULL, 
                            properties string
                        );'''

RES_CREATE_TABLE = '''CREATE TABLE IF NOT EXISTS estimates(
                            id integer PRIMARY KEY,
                            sc_name string NOT NULL, 
                            position string NOT NULL, 
                            velocity string NOT NULL, 
                            frame string NOT NULL, 
                            timestamp numeric NOT NULL, 
                            covariance blob NOT NULL,
                            properties string
                        );'''           


#======================
# Insert Strings
#======================
INSERT_TRUTH_STATE = '''INSERT INTO truth_states(position, velocity, frame, timestamp, properties) 
                        VALUES(?,?,?,?,?)'''


#======================
# Query Strings 
#======================
GET_LAST_TRUTH_STATE_TIMES = "SELECT timestamp FROM truth_states ORDER BY timestamp DESC LIMIT 1;"
GET_ALL_TRUTH_STATES = "SELECT position, velocity FROM truth_states ORDER BY timestamp DESC;"