import numpy as np
import matplotlib.pyplot as plt
import torch

from DynamicSystemData import DynamicSystemData, dynamic_system_cubed
from FlipFlop import FlipFlop

if __name__ == '__main__':
    # Simulate the system
    n_trials = 10
    time_span = (0, 5)
    timesteps = 300
    data_gen = DynamicSystemData(dynamic_system=dynamic_system_cubed)
    generated_data = data_gen.generate_data(n_trials, time_span, timesteps)

    n_hidden = 16
    rnn_type = "griffin-recurrent-block"
    results_filename = "models/dynamic_system_results.pth"

    n_train = 512
    n_valid = 128
    batch_size = 128
    train_data = data_gen.generate_data(n_trials=n_train, time_span=time_span, n_timesteps=timesteps)
    valid_data = data_gen.generate_data(n_trials=n_valid, time_span=time_span, n_timesteps=timesteps)

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
    save_results = False
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
