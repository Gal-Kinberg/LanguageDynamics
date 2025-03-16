'''
examples/torch/run_FlipFlop.py
Written for Python 3.8.17 and Pytorch 2.0.1
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PATH_TO_FIXED_POINT_FINDER = '../../'
PATH_TO_HELPER = '../helper/'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
sys.path.insert(0, PATH_TO_HELPER)

from FlipFlop import FlipFlop
from fixed_point_finder.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from FlipFlopData import FlipFlopData
from fixed_point_finder.plot_utils import plot_fps

def train_FlipFlop(results_filename: str = 'flipflop_results.pth', save_results: bool = True,
                   use_existing_results: bool = True,):
    ''' Train an RNN to solve the N-bit memory task.

        Args:
            None.

        Returns:
            model: FlipFlop object.

                The trained RNN model.

            valid_predictions: dict.

                The model's predictions on a set of held-out validation trials.
    '''

    exising_file_found = None
    # Try to load saved model and predictions if available
    if use_existing_results:
        try:
            loaded_data = torch.load(results_filename)
            model = loaded_data['model']
            valid_predictions = loaded_data['valid_predictions']
            exising_file_found = True
            print(f"Loaded model and predictions from {results_filename}.")
        except FileNotFoundError:
            exising_file_found = False
            print(f"No saved file found. Training a new model...")
        # Train the model and obtain predictions
    if not use_existing_results or (use_existing_results and exising_file_found is False):
        # Data specifications
        n_bits = 2
        n_time = 300
        n_train = 512
        n_valid = 128

        # Model hyperparameters
        n_hidden = 16
        batch_size = 128
        # rnn_type = 'tanh' # see note below
        rnn_type = 'griffin-recurrent-block' # see note below

        # Note: 'gru' should work in principle, and in the TF example it certainly does.
        # However, in this Pytorch example, fixed point finding in a GRU is not working
        # as expected.

        data_gen = FlipFlopData(n_bits=n_bits, n_time=n_time)
        train_data = data_gen.generate_data(n_trials=n_train)
        valid_data = data_gen.generate_data(n_trials=n_valid)

        model = FlipFlop(
            n_inputs=n_bits, # d
            n_hidden=n_hidden, # h
            n_outputs=n_bits, # d
            rnn_type=rnn_type)

        # learning_rate = 1./np.sqrt(batch_size)
        learning_rate = 1./np.sqrt(batch_size)
        losses, grad_norms = model.train(train_data, valid_data,
                                         learning_rate=learning_rate,
                                         batch_size=batch_size)
        valid_predictions = model.predict(valid_data)
        # Save the model and predictions for later
        if save_results:
            torch.save({'model': model, 'valid_predictions': valid_predictions}, results_filename)
            print(f"Model and predictions saved to {results_filename}.")

    return model, valid_predictions


def find_fixed_points(model, valid_predictions):
    ''' Find, analyze, and visualize the fixed points of the trained RNN.

    Args:
        model: FlipFlop object.

            Trained RNN model, as returned by train_FlipFlop().

        valid_predictions: dict.

            Model predictions on validation trials, as returned by
            train_FlipFlop().

    Returns:
        None.
    '''

    NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
    N_INITS = 1024 # The number of initial states to provide

    n_bits = valid_predictions['output'].shape[2]

    '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
    descriptions of available hyperparameters.'''
    fpf_hps = {
        'max_iters': 1000,
        'lr_init': 1.,
        'outlier_distance_scale': 10.0,
        'verbose': True,
        'super_verbose': True}

    # Set up the fixed point finder
    fpf = FixedPointFinder(model.rnn, **fpf_hps)

    '''Draw random, noise corrupted samples of those state trajectories
    to use as initial states for the fixed point optimizations.'''
    # valid_initial_states should be example trajectories of shape bxtxh (h is the full hidden state)
    valid_initial_states = valid_predictions['cache'] if model.rnn_type == 'griffin-recurrent-block' else valid_predictions['hidden']
    initial_states = fpf.sample_states(valid_initial_states,
                                       n_inits=N_INITS,
                                       noise_scale=NOISE_SCALE)

    # Study the system in the absence of input pulses (e.g., all inputs are 0)
    inputs = np.zeros([1, n_bits])

    # Run the fixed point finder
    unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

    # Visualize identified fixed points with overlaid RNN state trajectories
    # All visualized in the 3D PCA space fit the example RNN states.
    fig = plot_fps(unique_fps, valid_initial_states,
                   plot_batch_idx=list(range(30)),
                   plot_start_time=10)

    return unique_fps


def create_constant_flipflop_data(desired_inputs: np.ndarray, n_time = 300, active_fraction=0.5):
    '''Creates FlipFlopData object and generates constant data.'''
    n_bits = desired_inputs.shape[1]
    data_gen = FlipFlopData(n_bits=n_bits, n_time=n_time)
    constant_data = data_gen.generate_constant_data(desired_inputs, active_fraction=active_fraction)
    return constant_data



def plot_neurons(result_to_plot_txd, variable_name: str):
    # Plot each column of constant_predictions['cache'][0] on the same figure
    plt.figure(figsize=(10, 6))
    for i in range(result_to_plot_txd.shape[1]):
        plt.plot(result_to_plot_txd[:, i], label=f'Column {i}')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(f'Plot of Each Column in {variable_name}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    do_find_fixed_points = False

    # Step 1: Train an RNN to solve the N-bit memory task
    results_filename = 'models/flipflop_results.pth'
    save_results = False
    use_existing_results = True
    model, valid_predictions = train_FlipFlop(results_filename, save_results, use_existing_results)

    # STEP 2: Find, analyze, and visualize the fixed points of the trained RNN
    if do_find_fixed_points:
        unique_fps = find_fixed_points(model, valid_predictions)

    # test constant data
    n_time = 1000
    active_fraction = 0.1
    desired_inputs = np.array([[0, 0], [-1, -1], [1, -1], [-1, 1]])
    constant_data = create_constant_flipflop_data(desired_inputs, n_time=n_time, active_fraction=active_fraction)
    constant_predictions = model.predict(constant_data)

    variable_to_plot = 'hidden'
    plot_neurons(constant_predictions[f'{variable_to_plot}'][0], f'{variable_to_plot}')

    print('Entering debug mode to allow interaction with objects and figures.')
    print('You should see a figure with:')
    print('\tMany blue lines approximately outlining a cube')
    print('\tStable fixed points (black dots) at corners of the cube')
    print('\tUnstable fixed points (red lines or crosses) '
          'on edges, surfaces and center of the cube')
    print('Enter q to quit.\n')
    # pdb.set_trace()