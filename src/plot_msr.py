import json 
import sqlite3
from sql.queries import GetMsr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def get_stns(conn): 
    curs = conn.cursor()
    curs.execute('SELECT stn_id from stations;')
    query = curs.fetchall()
    curs.close()
    
    stn_dict = {int(stn_id):{"range":[], "range_rate":[], "times":[]} for (stn_id, ) in query}

    return stn_dict

def get_msrs(conn):
    curs = conn.cursor()
    curs.execute(GetMsr.ALL_MSRS)
    query = curs.fetchall()
    curs.close()

    return query


if __name__ == "__main__":
    conn = sqlite3.connect('./base_sim/sim1_msr.db') 

    fig, ax = plt.subplots(2, 1, sharex='all')
    stn_dict = get_stns(conn)
    msrs = get_msrs(conn)
    
    for msr in msrs: 
        msr_vals = json.loads(msr[1])
        stn_dict[msr[0]]["range"].append(msr_vals[0])
        #stn_dict[msr[0]]["range_rate"].append(msr_vals[1])
        stn_dict[msr[0]]["times"].append(msr[2])

    stn_labels = {
        11: "Goldstone", 
        35: "Canberra",
        54: "Madrid"
    }

    for stn in stn_dict: 
        ax[0].scatter(stn_dict[stn]["times"], stn_dict[stn]["range"], label=stn_labels[stn], s=0.8)
        #ax[1].scatter(stn_dict[stn]["times"], stn_dict[stn]["range_rate"], label=stn_labels[stn], s=0.8)

    ax[1].set_xlabel("Non-Dimensional CR3BP Time")
    ax[1].set_ylabel("CR3BP Range-Rate")
    ax[0].set_ylabel("CR3BP Range")
    ax[1].grid('on')
    ax[0].grid('on')
    ax[0].legend(title="Station IDs", loc='lower right')
    ax[0].set_title("True Measurements of spacecraft Range and Range-Rate")
    plt.show()