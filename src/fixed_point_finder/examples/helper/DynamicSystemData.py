import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class DynamicSystemData:

    def __init__(self, dynamic_system=None):
        """

        """

        self.dynamic_system = dynamic_system

    def generate_data(self, n_trials: int = 100, time_span = (0,10), n_timesteps: int = 300):
        """
        Generates time-series data for a dynamic system by simulating multiple
        trials over a specified time span. Each trial starts with a randomly
        sampled initial state and computes the trajectory of the system over
        discretized timesteps.

        :param n_trials: The number of trials (simulations) to run.
        :type n_trials: int
        :param time_span: A tuple specifying the start and end time
            (e.g., (0, 10)) for the simulation.
        :type time_span: tuple[float, float]
        :param n_timesteps: The number of timesteps to discretize the
            simulation time span into.
        :type n_timesteps: int
        :return: A tuple containing four items:
            - t: A numpy array of time points corresponding to the
              discretized timesteps within the given time span.
            - x_data_bxtxd: A numpy array containing the simulated data for
              each trial, where the first dimension indexes the trials, and
              the second dimension indexes the timestep.
            - initial_states_bx1: A numpy array of the randomly sampled initial
              states used as initial conditions for each trial.
            - targets_bxtxd: A numpy array like x_data_bxtxd but shifted by one
              timestep to represent the predicted next value. The last
              timestep of each trial is excluded.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """

        # Create a time array for the given number of timesteps
        t = np.linspace(time_span[0], time_span[1], n_timesteps)

        # Store all the results
        x_data_bxtxd = np.zeros((n_trials, n_timesteps, 1))

        # sample initial states
        initial_states_bx1 = np.random.randn(n_trials)

        # run simulations for each trial
        for trial in range(n_trials):
            initial_state = initial_states_bx1[trial]

            # run the simulation
            sol = solve_ivp(self.dynamic_system, time_span, [initial_state], t_eval=t, method='RK45')

            # save the results
            x_data_bxtxd[trial, :, 0] = sol.y[0]

        # Create the "targets_bxtxd" vector by shifting x_data_bxtxd by one timestep
        targets_bxtxd = x_data_bxtxd[:, 1:, :]

        return {
            "inputs": x_data_bxtxd[:, :-1, :].astype(np.float32),
            "targets": targets_bxtxd.astype(np.float32),
            "initial_states": initial_states_bx1,
            "t": t[:-1]
        }


def dynamic_system_cubed(t, x):
    return x - x ** 3