import numpy as np
import matplotlib.pyplot as plt
import torch

from DynamicSystemData import DynamicSystemData, dynamic_system_cubed
from FlipFlop import FlipFlop

from fixed_point_finder.FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from fixed_point_finder.plot_utils import plot_fps


def find_autoregressive_fixed_points(model, valid_predictions,
                      NOISE_SCALE=0.5,
                      N_INITS=1024):
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

    n_bits = valid_predictions['output'].shape[2]

    '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
    descriptions of available hyperparameters.'''
    fpf_hps = {
        'tol_q': 1e-12,
        'tol_dq': 0,
        'max_iters': 100000,
        'lr_init': 1.,
        'outlier_distance_scale': 10.0,
        'verbose': True,
        'super_verbose': True}

    # Set up the fixed point finder
    fpf = FixedPointFinder(model, autoregressive_mode=True,
                           lr_patience=10,
                           lr_factor=0.95,
                           **fpf_hps)

    '''Draw random, noise corrupted samples of those state trajectories
    to use as initial states for the fixed point optimizations.'''
    # valid_initial_states should be example trajectories of shape bxtxh (h is the full hidden state)
    # valid_initial_states = valid_predictions['cache'] if model.rnn_type == 'griffin-recurrent-block' else \
    #     valid_predictions['hidden']
    valid_initial_states = np.concat([valid_predictions['cache'], valid_predictions['y']], axis=2)
    initial_states = fpf.sample_states(valid_initial_states,
                                       n_inits=N_INITS,
                                       noise_scale=NOISE_SCALE)

    # Study the system in the absence of input pulses (e.g., all inputs are 0)
    inputs = np.zeros([1, n_bits])

    # Run the fixed point finder
    unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

    # Visualize identified fixed points with overlaid RNN state trajectories
    # All visualized in the 3D PCA space fit the example RNN states.
    fig, pca = plot_fps(unique_fps, valid_initial_states,
                        plot_batch_idx=list(range(30)),
                        plot_start_time=10)

    return unique_fps, pca


if __name__ == '__main__':
    # Simulate the system
    n_trials = 10
    time_span = (0, 5)
    timesteps = 300
    data_gen = DynamicSystemData(dynamic_system=dynamic_system_cubed)
    generated_data = data_gen.generate_data(n_trials, time_span, timesteps)

    # Configuration
    use_existing_results = True
    save_results = False

    n_hidden = 16
    rnn_type = "griffin-recurrent-block"
    results_filename = "models/cubed_dynamic_system_results.pth"

    n_train = 512
    n_valid = 128
    batch_size = 128
    train_data = data_gen.generate_data(n_trials=n_train, time_span=time_span, n_timesteps=timesteps)
    valid_data = data_gen.generate_data(n_trials=n_valid, time_span=time_span, n_timesteps=timesteps)

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

        model = FlipFlop(
            n_inputs=1,  # d
            n_hidden=n_hidden,  # h
            n_outputs=1,  # d
            rnn_type=rnn_type)

        # learning_rate = 1./np.sqrt(batch_size)
        learning_rate = 1. / np.sqrt(batch_size)
        losses, grad_norms = model.train(train_data, valid_data,
                                         learning_rate=learning_rate,
                                         batch_size=batch_size)
        valid_predictions = model.predict(valid_data)

        # Save the model and predictions for later
        if save_results:
            torch.save({'model': model, 'valid_predictions': valid_predictions}, results_filename)
            print(f"Model and predictions saved to {results_filename}.")

    # Plot the results for a few trials
    plt.figure(figsize=(8, 6))
    for i in range(n_valid):
        plt.plot(valid_data['t'], valid_data["targets"][i, :], label=f'Trial {i + 1}')
    plt.xlabel('Time')
    plt.ylabel('x(t)')
    plt.title('Simulation of dx/dt = x - x^3')
    plt.legend()
    plt.grid()
    plt.show()

    # Fixed Point Finding
    NOISE_SCALE = 0.5  # Standard deviation of noise added to initial states
    N_INITS = 1024  # The number of initial states to provide

    unique_fps, pca = find_autoregressive_fixed_points(model, valid_predictions, NOISE_SCALE, N_INITS)