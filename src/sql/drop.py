#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/sql_schema.py
Description: Defines sqlite drop commands for table removal or replacement
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
class DropMsr(object):
    """ Contianer object for drop commands on Measurement database """
    STN_IF_EXISTS = "DROP TABLE IF EXISTS stations;"
    MSR_IF_EXISTS = "DROP TABLE IF EXISTS measurements;"