#=========================
# Test Functions
#=========================    
# Runs test of the dynamics
if __name__ == "__main__":
    from numdifftools import Jacobian
    from simulation.dynamics_partials import cr3bp_jacobian
    from simulation.cr3bp import CR3BPSystem
    from simulation.constants import MU_EARTH_MOON
    from functools import partial
    from random import randint
    import numpy as np

    cr3bp_system = CR3BPSystem(MU_EARTH_MOON)
    TOL = 1e-6
    NUM_TESTS = 1000

    for _ in range(NUM_TESTS):
        test_state = [randint(0, 100) * 0.33 for _ in range(6)]
        partial_deriv = partial(cr3bp_system.derivative, 0.0)
        jacobian_true = Jacobian(partial_deriv)(test_state)
        jacobian_est = cr3bp_jacobian(MU_EARTH_MOON, test_state)

        for i in range(6):
            for j in range(6):
                try:
                    if abs(jacobian_est[i][j] - jacobian_true[i][j]) > TOL:
                        with np.printoptions(precision=3, suppress=True):
                            print("ELEMENT [{}][{}] Does not match".format(i,j))
                            print("EST: {}".format(jacobian_est[i][j]))
                            print("TRUE: {}".format(jacobian_true[i][j]))
                            print("===== True ======= \n", jacobian_true, "\n ===============")
                            print("===== Est ======== \n", jacobian_est, "\n ===============")
                            raise AssertionError("True and Estimated Jacobians do not match")
                except Exception as e: 
                    print("FAILED with | {}".format(str(e)))
                    print("EST", jacobian_est)
                    print("TRUE", jacobian_true)
                    exit()

                
    print("{} Tests Passed".format(NUM_TESTS))

    exit()