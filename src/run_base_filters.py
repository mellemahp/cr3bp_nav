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
import json

# local
from simulation.filters import EKFilter, UKFilter
from simulation.cr3bp import CR3BPSystem
from simulation.constants import(
    MU_EARTH_MOON, 
    NON_DIM_DIST_TO_DIM, 
    NON_DIM_TIME_TO_DIM
) 
from simulation.measurements import R3Msr, Range
from simulation.stations import CR3BPEarthStn

from sql.queries import GetMsr
from sql.schema import CreateEst
from sql.drop import DropEst
from sql.insert import InsertEst

# === End Imports ===

def load_msrs(conn_msr, stn_dict): 
    """ get all measurements and load into msr objects 

    """
    print("Loading measurements...")
    curs = conn_msr.cursor()
    curs.execute(GetMsr.ALL_MSRS)
    query = curs.fetchall()
    msr_list = [globals()[m[3]].from_db(m, stn_dict) for m in query]
    curs.close()
    print("Measurements loaded")

    return msr_list


def load_stns(conn_msr):
    """ Loads all station objects from database"""

    print("Loading Stations...")
    conn_msr.row_factory = sqlite3.Row
    curs = conn_msr.cursor()
    curs.execute('SELECT * from stations')
    query_res = [dict(q) for q in curs.fetchall()]
    stn_dict = {s['stn_id']: CR3BPEarthStn.from_db_object(s) for s in query_res}
    print("Stations Loaded")

    return stn_dict


def filter_params(config): 
    """ Creates a filter 

    """
    cr3bp_system = CR3BPSystem(MU_EARTH_MOON)

    # get truth state
    istate_position = [float(x) for x in config["satellite"]["istate"]["position"]]
    istate_velocity = [float(x) for x in config["satellite"]["istate"]["velocity"]]
    istate_true = istate_position + istate_velocity

    # find offsets in position and velocity
    offset_position_km = [float(x) for x in config["filters"]["perturbation"]["position"]]
    offset_velocity_km = [float(x) for x in config["filters"]["perturbation"]["velocity"]]
    offset_position_nd = [v * (1 / NON_DIM_DIST_TO_DIM) for v in offset_position_km]
    offset_velocity_nd = [v * (NON_DIM_TIME_TO_DIM / NON_DIM_DIST_TO_DIM) for v in offset_velocity_km]
    offsets = offset_position_nd + offset_velocity_nd

    istate = np.add(istate_true, offsets)

    # set initial guess for covariance
    apriori_pos = [config["filters"]["icov"]["position"] for _ in range(3)]
    apriori_vel = [config["filters"]["icov"]["velocity"] for _ in range(3)]
    apriori_vec = apriori_pos + apriori_vel
    apriori_matrix = np.diag(apriori_vec)

    return istate, apriori_matrix, cr3bp_system

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
    estimates_db_path = (
        cwd
        + "/"
        + config["results"]["folder"]
        + "/"
        + config["results"]["estimates"]
    )
    
    return (sqlite3.connect(msr_db_path), sqlite3.connect(estimates_db_path))

def setupDB(conn_est, restart): 
    """ Sets up estimate database """
    curs = conn_est.cursor()
    if restart:
        print("Dropping existing tables")
        curs.execute(DropEst.EST_IF_EXISTS)
        print("Re-building Estimates tables")
    
    curs.execute(CreateEst.EST_TABLE)
    curs.close()


def run_filters(istate, apriori, cr3bpsys, msrs): 
    """ """
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
    print("Running UKF...")
    begin_time_ukf = time.time()
    try:
        ukf.run()
    except Exception as e: 
        print(e)
        print(ukf)
        print(ukf.estimates)
        print(ukf.cov_list[-1])
        raise e
    end_time_ukf = time.time() - begin_time_ukf
    print("UKF finished in {} sec".format(end_time_ukf))

    print("Running EKF...")
    begin_time_ekf = time.time()
    try: 
        ekf.run()
    except Exception as e: 
        print(e)
        print(ekf)
        print(ekf.estimates)
        print(ekf.cov_list[-1])
        raise e  
    end_time_ekf = time.time() - begin_time_ekf
    print("EKF finished in {} sec".format(end_time_ekf))

    return ekf, ukf

def dump_msrs_to_db(filter, conn_est): 
    print("Writing estimates for {} to db...".format(filter.__class__.__name__))
    curs = conn_est.cursor()
    for time, state_est, cov, resid in zip(filter.times, filter.estimates, filter.cov_list, filter.residuals): 
        curs.execute(
            InsertEst.EST,
            (
                filter.__class__.__name__, 
                time, 
                json.dumps(state_est.tolist()),
                json.dumps(cov.tolist()), 
                json.dumps(resid.tolist())
            ),
        )
    conn_est.commit()
    curs.close()
    print("Batch of estimates written to db")

def main(args):
    """ Main Method for managing a simulation """
    with open(args.config) as f:
        config = toml.load(f)

    # sets up sqlite database if they do not exist or if restart flag thrown
    (conn_msr, conn_est) = connect_to_db(config)
    setupDB(conn_est, args.r)
    stn_dict = load_stns(conn_msr)
    msrs = load_msrs(conn_msr, stn_dict)
    istate, apriori, cr3bpsys = filter_params(config)
    ekf, ukf = run_filters(istate, apriori, cr3bpsys, msrs)
    dump_msrs_to_db(ukf, conn_est)
    dump_msrs_to_db(ekf, conn_est)


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
    main(args)

    ### exit ###
    end_time = time.time() - begin_time
    print("---- Base Filter Run Complete: {} seconds ----".format(end_time))
    exit()
