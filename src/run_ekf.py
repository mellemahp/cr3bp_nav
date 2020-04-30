#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/run_sim.py
Description: Command line tool for running an EKF filter on the CR3BP Data
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
# === Begin Imports ===
# third party
import numpy as np

# standard library
import time
import argparse
import toml
import os
import sqlite3

# local
from simulation.filters import EKFilter, UKFilter
from simulation.cr3bp import CR3BPSystem
from simulation.constants import MU_EARTH_MOON
from simulation.measurements import R3Msr
from simulation.stations import CR3BPEarthStn

from sql.queries import GetMsr

# === End Imports ===

def load_msrs(conn_msr, stn_dict): 
    """ get all measurements and load into msr objects 

    """
    curs = conn_msr.cursor()
    curs.execute(GetMsr.ALL_MSRS)
    query = curs.fetchall()
    msr_list = [globals()[m[3]].from_db(m, stn_dict) for m in query]
    curs.close()

    return msr_list


def filter_params(config): 
    """ Creates a filter 

    """
    cr3bp_system = CR3BPSystem(MU_EARTH_MOON)

    # set initial guess for state
    istate_position = [float(x) for x in config["filters"]["istate"]["position"]]
    istate_velocity = [float(x) for x in config["filters"]["istate"]["velocity"]]
    istate = istate_position + istate_velocity

    # set initial guess for covariance
    apriori_pos = [config["filters"]["icov"]["position"] for _ in range(3)]
    apriori_vel = [config["filters"]["icov"]["velocity"] for _ in range(3)]
    apriori_vec = apriori_pos + apriori_vel
    apriori_matrix = np.diag(apriori_vec)

    return istate, apriori_matrix, cr3bp_system


def load_stns(conn_msr):
    """ Loads all station objects """
    # extract stations from DB
    conn_msr.row_factory = sqlite3.Row
    curs = conn_msr.cursor()
    curs.execute('SELECT * from stations')
    query_res = [dict(q) for q in curs.fetchall()]
    stn_dict = {s['stn_id']: CR3BPEarthStn.from_db_object(s) for s in query_res}

    return stn_dict


def connect_to_db(config):
    """Connects to an SQL measurement database 

    """ 
    cwd = os.getcwd()
    msr_db_path = (
        cwd
        + "/"
        + config["results"]["folder"]
        + "/"
        + config["results"]["measurements"]
    )
    
    return sqlite3.connect(msr_db_path)


def main(args):
    """ Main Method for managing a simulation """
    with open(args.config) as f:
        config = toml.load(f)

    # sets up sqlite database if they do not exist or if restart flag thrown
    conn_msr = connect_to_db(config)
    print("Loading Stations and measurements...")
    stn_dict = load_stns(conn_msr)
    msrs = load_msrs(conn_msr, stn_dict)
    print("Stations and measurements loaded.")
    istate, apriori, cr3bpsys = filter_params(config)
    ekf = EKFilter(
        istate, 
        msrs, 
        apriori, 
        cr3bpsys.derivative, 
        cr3bpsys.jacobian  
    )
    ukf = UKFilter(
        istate, 
        msrs, 
        apriori, 
        cr3bpsys.derivative
    )
    print("Starting measurement processing")
    try:
        ukf.run()
    except Exception as e: 
        print(e)
        print(ukf)
        print(ekf.estimates)
        print(ekf.cov_list[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an EKF on simulation data")
    parser.add_argument(
        "config", metavar="C", help="Simulation config file in TOML format"
    )
    parser.add_argument(
        "-r", action="store_true", help="Restart the simulation. This replaces old data"
    )
    args = parser.parse_args()

    ### run main program ###
    begin_time = time.time()
    print("Starting EKF Filtering...")
    main(args)

    ### exit ###
    end_time = time.time() - begin_time
    print("---- EKF Run Complete: {} seconds ----".format(end_time))
    exit()
