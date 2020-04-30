#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/run_sim.py
Description: Simple command0line tool for running CR3BP simulations
Project: Advanced State Estimation Final Project
Author: Hunter Mellema
Date: April 2020
"""
# === Begin Imports ===
# third party
import numpy as np
from scipy.integrate import solve_ivp

# std library
import toml
import json
import time
import argparse
import sqlite3
import os
from os import path
from copy import deepcopy

# local imports
from sql.schema import CreateMsr
from sql.insert import InsertMsr
from sql.queries import GetMsr

from simulation.cr3bp import CR3BPSystem
from simulation.constants import(
    MU_EARTH_MOON, NON_DIM_TIME_TO_DIM, NON_DIM_DIST_TO_DIM
) 

# === End Imports ===

# ===================================
# Folder, Database, File Management
# ===================================
def folder_setup(config):
    """ Creates a new results folder if one does not alread exist for the simulation
    """
    if not path.exists("./" + config["results"]["folder"]):
        cwd = os.getcwd()
        results_path = cwd + "/" + config["results"]["folder"]
        print(
            "No results folder found. Creating results folder: \n {}".format(
                results_path
            )
        )
        os.mkdir(results_path)


def database_setup(config, restart):
    """Creates new results and measurements databases if ones do not exist or if restart flag is thrown

    """
    cwd = os.getcwd()
    msr_db_path = (
        cwd
        + "/"
        + config["results"]["folder"]
        + "/"
        + config["results"]["measurements"]
    )
    res_db_path = (
        cwd + "/" + config["results"]["folder"] + "/" + config["results"]["estimates"]
    )

    if restart:
        try:
            print("Removing old databases...")
            os.remove(msr_db_path)
            os.remove(res_db_path)
            print("Old databases removed")
        except FileNotFoundError:
            print("No databases found to replace. Continuing...")


    if not path.exists(msr_db_path):
        print("Creating new measurement database...")

        # msr db setup
        conn_msr = sqlite3.connect(msr_db_path)
        curs_msr = conn_msr.cursor()
        curs_msr.execute(CreateMsr.TRUTH_STATES_TABLE)
        curs_msr.close()

        print("New Databases created")

    conn_msr = sqlite3.connect(msr_db_path)

    return conn_msr


# ===================================
# Measurement Simulation
# ===================================
def get_simulation_times(config):
    """ Produces a list of non-dimensional times to evaluate 

    """
    if config["simulation"]["dimensional"]:
        start_time = config["simulation"]["start"] / NON_DIM_TIME_TO_DIM
        end_time = config["simulation"]["end"] / NON_DIM_TIME_TO_DIM
        msr_timestep = config["simulation"]["msr_timestep"] / NON_DIM_TIME_TO_DIM
    else:
        start_time = config["simulation"]["start"]
        end_time = config["simulation"]["end"]
        msr_timestep = config["simulation"]["msr_timestep"]

    return np.arange(start_time, end_time, msr_timestep)


def simulate_trajectory(config, times, conn_msr):
    """Uses the LSODA integrator to propagate spacecraft trajectory 
    
    """
    # check what last time in the database is
    c = conn_msr.execute(GetMsr.LAST_TRUTH_STATE_TIMES)
    query_res = c.fetchall()
    last_time_in_db = query_res[0] if query_res else None

    if last_time_in_db and last_time_in_db >= times[-1]: 
        print("Using previously propagated data")
        c = conn_msr.execute(GetMsr.ALL_TRUTH_STATES)
        return [
            json.loads(pos) + json.loads(vel) for pos, vel in 
            c.fetchall()
        ]

    # databse cursor
    curs = conn_msr.cursor()

    # set up system and initial state
    cr3bp_system = CR3BPSystem(MU_EARTH_MOON)
    istate_position = [float(x) for x in config["satellite"]["istate"]["position"]]
    istate_velocity = [float(x) for x in config["satellite"]["istate"]["velocity"]]
    istate = istate_position + istate_velocity

    if config['satellite']['istate']['dimensional']:
        istate = [x / NON_DIM_DIST_TO_DIM for x in istate]

    states = [istate]
    print("Propagating Satellite Trajectory...")
    sol = solve_ivp(
        cr3bp_system.derivative, 
        [times[0], times[-1]], 
        states[-1], 
        method="LSODA", 
        rtol=1e-6, 
        atol=1e-9,
        dense_output=True
    )

    try:
        for time_next in times[1:]:
            sol_val = sol.sol(time_next).tolist()

            # attempt to write to database
            curs.execute(
                InsertMsr.TRUTH_STATE,
                (
                    json.dumps(sol_val[:3]),
                    json.dumps(sol_val[3:]),
                    config["simulation"]["reference_frame"],
                    time_next,
                    json.dumps(config["satellite"]["properties"]),
                ),
            )
            conn_msr.commit()
            states.append(sol_val)

    finally:
        curs.close()

    return states


def main(args):
    """ Main Method for managing a simulation """
    with open(args.config) as f:
        config = toml.load(f)

    # sets up folder if it does not exist
    folder_setup(config)

    # sets up sqlite database if they do not exist or if restart flag thrown
    conn_msr = database_setup(config, args.r)

    # run trajectory and measurement simulation
    times = get_simulation_times(config)
    simulate_trajectory(config, times, conn_msr)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CR3BP simulation")
    parser.add_argument(
        "config", metavar="C", help="Simulation config file in TOML format"
    )
    parser.add_argument(
        "-r", action="store_true", help="Restart the simulation. This replaces old data"
    )
    args = parser.parse_args()

    ### run main program ###
    begin_time = time.time()
    print("Starting Simulation...")
    main(args)

    ### exit ###
    end_time = time.time() - begin_time
    print("---- Trajectory Simulation Complete: {} seconds ----".format(end_time))
    exit()
