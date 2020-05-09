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
from sql.drop import DropMsr

from simulation.cr3bp import CR3BPSystem
from simulation.stations import CR3BPEarthStn
from simulation.constants import MU_EARTH_MOON, NON_DIM_TIME_TO_DIM, NON_DIM_DIST_TO_DIM

# === End Imports ===

# ===================================
# Folder, Database, File Management
# ===================================
def folder_setup(config):
    """ Creates a new results folder if one does not already exist for the simulation
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


def msr_database_setup(conn_msr, restart):
    """Creates new results and measurements databases if ones do not exist or if restart flag is thrown

    """
    curs = conn_msr.cursor()
    if restart:
        print("Dropping existing tables")
        curs.execute(DropMsr.STN_IF_EXISTS)
        curs.execute(DropMsr.MSR_IF_EXISTS)
        print("Re-building Stations and Measurement tables")
    
    curs.execute(CreateMsr.STN_TABLE)
    curs.execute(CreateMsr.MSR_TABLE)
    curs.close()

    return 0


def create_stations(config, conn_msr, restart):
    """ Establishes Stations in Database"""
    if restart == True: 
        print("Adding Stations To database...")
        stn_template = {
            'stn_id': None,
            'stn_name': None,
            'latitude': None,
            'longitude': None,
            'elevation': None,
            'el_mask': None, 
            'measurements': None,
            'covariance': None, 
            'mu': MU_EARTH_MOON
        }
        
        for stn in [stn for stn in config['stations'] if not stn == 'defaults']: 
            stn_data = deepcopy(stn_template)
            stn_data['stn_name'] = stn
            for val in [val for val in stn_template if not val in ['stn_name', 'mu']]: 
                try: 
                    data = config['stations']['defaults'][val]
                except KeyError: 
                    data = config['stations'][stn][val]

                stn_data[val] = data

            curs = conn_msr.cursor()
            curs.execute(
                InsertMsr.STN,
                (
                    stn_data['stn_id'],
                    stn_data['stn_name'], 
                    json.dumps(stn_data['covariance']),
                    stn_data['latitude'], 
                    stn_data['longitude'], 
                    stn_data['elevation'], 
                    stn_data['el_mask'], 
                    json.dumps(stn_data['measurements']), 
                    stn_data['mu']
                ),
            )
        print("Stations added to db")
    else: 
        print("Using Pre-stored stations")

    # extract stations from DB
    conn_msr.row_factory = sqlite3.Row
    curs = conn_msr.cursor()
    curs.execute('SELECT * from stations')
    query_res = [dict(q) for q in curs.fetchall()]
    station_list = [CR3BPEarthStn.from_db_object(s) for s in query_res]

    return station_list


def simulate_measurements(config, times, states, station_list, conn_msr): 
    """ Simulates measurements for stations """
    msr_list = []
    print("Generating measurements...")

    for time, state in zip(times, states): 
        for station in station_list: 
            msr_list.append(station.gen_msr(state, time, "Range"))
    
    print("Writing measurements to db")
    # attempt to write to database, removing any null msrs
    for msr in [m for m in msr_list if m is not None]:
        # add noise! 
        msr.add_white_noise([eval(c) for c in msr.cov])

        curs = conn_msr.cursor()
        curs.execute(
            InsertMsr.MSR,
            (
                int(msr.stn), 
                json.dumps(msr.msr.tolist()[0]),
                msr.time,
                msr.__class__.__name__ ,
                json.dumps(msr.cov),
            ),
        )
    conn_msr.commit()

    return 0

def get_traj(conn_msr): 
    """
    """
    print("Extracting states and times...")
    curs = conn_msr.cursor()
    curs.execute(GetMsr.ALL_TRUTH_STATE_AND_TIMES)
    res = curs.fetchall()

    if res is []: 
        raise ValueError("No Trajectory states found")

    states = []
    times = []
    for (pos, vel, time) in res: 
        times.append(time)
        states.append(json.loads(pos) + json.loads(vel))

    print("States and times extracted.")

    return (states, times)

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
    msr_database_setup(conn_msr, args.r)

    # add stationbs
    stn_list = create_stations(config, conn_msr, args.r)
    states, times = get_traj(conn_msr)
    simulate_measurements(config, times, states, stn_list, conn_msr)


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
    print("---- Measurement Simulation Complete: {} seconds ----".format(end_time))
    exit()
