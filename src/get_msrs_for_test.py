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
from simulation.gmm import *
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

def connect_to_db():
    """Connects to an SQL measurement database 

    """ 
    cwd = os.getcwd()
    msr_db_path = "../base_sim/sim1_msr.db"
    estimates_db_path = "../base_sim/sim1_est.db"
    
    return (sqlite3.connect(msr_db_path), sqlite3.connect(estimates_db_path))

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

conn_msr, conn_est = connect_to_db()
stn_dict = load_stns(conn_msr)
msrs = load_msrs(conn_msr, stn_dict)
offset_position_km = [-10.0, 10.0, 5.0]
offset_velocity_km = [0.001, 0.001, 0.001]
offset_position_nd = [v * (1 / NON_DIM_DIST_TO_DIM) for v in offset_position_km]
offset_velocity_nd = [v * (NON_DIM_TIME_TO_DIM / NON_DIM_DIST_TO_DIM) for v in offset_velocity_km]
shift = np.array(offset_position_nd + offset_velocity_nd)
istate = np.array([-0.826, 0.0, 0.0, 0.0, 0.100, 0.1])
apriori = np.diag([0.00025, 0.00025, 0.00025, 0.01, 0.01, 0.01])
cr3bp_system = CR3BPSystem(MU_EARTH_MOON)


gmm = GMM(istate, msrs, apriori, cr3bp_system.derivative, cr3bp_system.jacobian)  