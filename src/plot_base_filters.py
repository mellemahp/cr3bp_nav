import json
from sql.queries import GetMsr, GetEst
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import argparse

def get_states(conn_msr): 
    curs = conn_msr.cursor()
    curs.execute(GetMsr.ALL_TRUTH_STATES)
    times = []
    states = []
    for q in curs.fetchall(): 
        states.append(json.loads(q[0]) + json.loads(q[1]))
        times.append(q[2])
        
    return (times, states)

def get_ests(conn_est, filter_name): 
    """ """ 
    curs = conn_est.cursor()
    curs.execute(GetEst.FILTER_ESTS, (filter_name,))
    covariances = []
    times = []
    states = []
    residuals = []
    for time, estimate, cov, resid in curs.fetchall():
        times.append(time)
        states.append(json.loads(estimate))
        covariances.append(np.array(json.loads(cov)))
        residuals.append(np.array(json.loads(resid)))
    curs.close()
    return times, states, covariances, residuals

def plot_pos(ax, times, ests, label): 
    """ """ 
    
    for idx in range(3): 
        vals = []
        for e in ests: 
            vals.append(e[idx])

        ax[0,idx].plot(times, vals, '-', markersize=0.5, label=label)

def plot_vel(ax, times, ests, label): 
    """ """ 
    
    for idx in range(3): 
        vals = []
        for e in ests: 
            vals.append(e[idx + 3])

        ax[1,idx].plot(times, vals, '-', markersize=0.5, label=label)

def plot_pos_vel(times_truth, states_truth, times_ukf, ests_ukf, times_ekf, ests_ekf): 
        fig, ax = plt.subplots(2, 3, sharey='row', sharex='col')
        
        plot_pos(ax, times_ekf, ests_ekf, "EKF")
        plot_pos(ax, times_ukf, ests_ukf, "UKF")
        plot_pos(ax, times_truth, states_truth, "Truth")
        plot_vel(ax, times_ekf, ests_ekf, "EKF")
        plot_vel(ax, times_ukf, ests_ukf, "UKF")
        plot_vel(ax, times_truth, states_truth, "Truth")
        ax[0,2].legend(loc='lower right')
        ax[1,1].set_xlabel("Time (non-dimensional)")
        ax[0,0].set_ylabel("Position (non-dim)")
        ax[1,0].set_ylabel("Velocity (non-dim)")
        ax[0,0].set_title("X-Axis")
        ax[0,1].set_title("Y-Axis")
        ax[0,2].set_title("Z-Axis")
        ax[0,0].set_ylim([-1, 1])
        ax[1,0].set_ylim([-2, 2])
        plt.show()

def plot_resids(times_ekf, resids_ekf, times_ukf, resids_ukf): 
    plt.plot(times_ekf, [x[0]**2 for x in resids_ekf], 'o', markersize=0.5, label="EKF")
    plt.plot(times_ukf, [x[0]**2 for x in resids_ukf], 'o', markersize=0.5, label="UKF")
    plt.xlabel("Time (non-dimensional)")
    plt.ylabel("Distance Measurement residual")
    plt.title("Squared residual plot")
    #plt.ylim([-0.0, 0.0001])
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.show()


def plot_cov(ax, times, covs, label): 
    for idx in range(3): 
        vals_cov = []        #for d in diffs: 
        for c in covs: 
            vals_cov.append(c[idx])

        ax[0,idx].plot(times, vals_cov, '-', markersize=0.5, label=label)
        ax[0,idx].set_yscale('log')

    for idx in range(3): 
        vals_cov = []        #for d in diffs: 
        for c in covs: 
            vals_cov.append(c[idx + 3])

        ax[1,idx].plot(times, vals_cov, '-', markersize=0.5, label=label)
        ax[1,idx].set_yscale('log')

def plot_diffs(ax, times, diffs, label): 
    for idx in range(3): 
        vals_diff = []        #for d in diffs: 
        for d in diffs: 
            vals_diff.append(d[idx])

        ax[0,idx].plot(times, vals_diff, '-', markersize=0.5, label=label)
        ax[0,idx].set_yscale('log')

    for idx in range(3): 
        vals_diff = []        #for d in diffs: 
        for d in diffs: 
            vals_diff.append(d[idx + 3])

        ax[1,idx].plot(times, vals_diff, '-', markersize=0.5, label=label)
        ax[1,idx].set_yscale('log')


def filter_times_states(time_truth, state_truth, times_filter, states_filter): 
    diffs = []
    times = []
    for idx, t in enumerate(times_filter):
        if t != 0.0:
            jdx = time_truth.index(t)
            diffs.append(np.abs(np.subtract(state_truth[jdx], states_filter[idx])))
            times.append(t)

    return times, diffs


def plot_covariance(times_ekf, covs_ekf, times_ukf, covs_ukf): 
    diags_ekf = [np.sqrt(np.abs(np.diag(cov))) for cov in covs_ekf]
    diags_ukf = [np.sqrt(np.abs(np.diag(cov))) for cov in covs_ukf]

    import matplotlib.style as style
    style.use('seaborn')
    fig, ax = plt.subplots(2, 3, sharey='row', sharex='col')
    # plot pos vals
    plot_cov(ax, times_ekf, diags_ekf, "EKF")
    plot_cov(ax, times_ukf, diags_ukf, "UKF")

    ax[0,0].set_ylabel("Position (non-dim)")
    ax[1,0].set_ylabel("Velocity (non-dim)")
    ax[0,0].set_title("X-Axis")
    ax[0,1].set_title("Y-Axis")
    ax[0,2].set_title("Z-Axis")

    plt.legend()
    plt.show()

def plot_diffs_interface(times_truth, states_truth, times_ukf, ests_ukf, times_ekf, ests_ekf): 
    times_ekf, diffs_ekf = filter_times_states(times_truth, states_truth, times_ekf, ests_ekf)
    times_ukf, diffs_ukf = filter_times_states(times_truth, states_truth, times_ukf, ests_ukf)

    import matplotlib.style as style
    style.use('seaborn')
    fig, ax = plt.subplots(2, 3, sharey='row', sharex='col')

    plot_diffs(ax, times_ekf, diffs_ekf, "EKF") 
    plot_diffs(ax, times_ukf, diffs_ukf, "UKF")

    ax[0,0].set_ylabel("Position (non-dim)")
    ax[1,0].set_ylabel("Velocity (non-dim)")
    ax[1,1].set_xlabel("Time (non-dimensional)")
    ax[0,0].set_title("X-Axis")
    ax[0,1].set_title("Y-Axis")
    ax[0,2].set_title("Z-Axis")
    plt.legend()
    plt.show()

def main(conn_est, conn_msr, args):
    times_truth, states_truth = get_states(conn_msr)
    times_ukf, ests_ukf, covs_ukf, resids_ukf = get_ests(conn_est, 'UKFilter')
    times_ekf, ests_ekf, covs_ekf, resids_ekf = get_ests(conn_est, 'EKFilter') 

    if args.p:
        plot_pos_vel(times_truth, states_truth, times_ukf, ests_ukf, times_ekf, ests_ekf)
    elif args.resid:
        plot_resids(times_ekf, resids_ekf, times_ukf, resids_ukf)
    elif args.cov: 
        plot_covariance(times_ekf, covs_ekf, times_ukf, covs_ukf)
    elif args.diff: 
        plot_diffs_interface(times_truth, states_truth, times_ukf, ests_ukf, times_ekf, ests_ekf)


if __name__ == "__main__":
    conn_msr = sqlite3.connect('./base_sim/sim1_msr.db') 
    conn_est = sqlite3.connect('./base_sim/sim1_est.db') 

    parser = argparse.ArgumentParser(description="Graph results from base filters")
    parser.add_argument(
        "-resid", action="store_true", help="Plot residuals"
    )
    parser.add_argument(
        "-p", action="store_true", help="Plot positions and velocities"
    )
    parser.add_argument(
        "-cov", action="store_true", help="Plot covariance"
    )
    parser.add_argument(
        "-diff", action="store_true", help="Plot absolute error"
    )
    args = parser.parse_args()

    main(conn_est, conn_msr, args)
