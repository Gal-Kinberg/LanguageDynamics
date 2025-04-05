import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA


def create_plot(data, N=None):
    """
    Creates an interactive Plotly figure for (N, 2) or (N, 3) shaped numpy arrays or torch tensors.
    The color of each point is based on its index in the input.

    Parameters:
        data (numpy.ndarray or torch.Tensor): Input data of shape (N, 2) or (N, 3).
    """
    # Convert torch tensor to numpy array if needed
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    # Validate the shape of the input
    if data.ndim != 2 or data.shape[1] not in [2, 3]:
        raise ValueError("Input data must have shape (N, 2) or (N, 3).")

    # Index values for coloring
    indices = np.arange(data.shape[0])
    if N is not None:
        indices = np.mod(indices, N)

    if data.shape[1] == 2:
        # 2D case
        x, y = data[:, 0], data[:, 1]
        fig = go.Figure(data=go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=6, color=indices, colorscale='Viridis', colorbar=dict(title="Index"))
        ))
        fig.update_layout(
            title="2D Scatter Plot",
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            template="plotly_dark"
        )
    elif data.shape[1] == 3:
        # 3D case
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=4, color=indices, colorscale='Viridis', colorbar=dict(title="Index"), opacity=0.8)
        ))
        fig.update_layout(
            title="3D Scatter Plot",
            scene=dict(
                xaxis_title="X-axis",
                yaxis_title="Y-axis",
                zaxis_title="Z-axis"
            ),
            template="plotly_dark"
        )

    fig.show()


def fit_pca(input_nxd, n_components=3):
    # input validation
    # Convert torch tensor to numpy array if needed
    if isinstance(input_nxd, torch.Tensor):
        input_nxd = input_nxd.numpy()

    # Validate the shape of the input
    if input_nxd.ndim == 3:
        # input is of shape bxtxd. reshape to b*txd
        input_nxd = input_nxd.reshape(input_nxd.shape[0] * input_nxd.shape[1], input_nxd.shape[2])
    elif input_nxd.ndim > 3:
        raise ValueError("Only inputs of shape NxD or BxNxD are supported.")

    pca = PCA(n_components=n_components)

    pca.fit(input_nxd)

    transformerd_input = pca.transform(input_nxd)

    return transformerd_input, pca


def plot_fps_plotly(fps,
                    state_traj=None,
                    plot_batch_idx=None,
                    plot_start_time=0,
                    plot_stop_time=None,
                    mode_scale=0.25,
                    fig=None,
                    output_colors=None):
    '''Plots a visualization and analysis of the unique fixed points.
    1) Finds a low-dimensional subspace for visualization via PCA. If
    state_traj is provided, PCA is fit to [all of] those RNN state
    trajectories. Otherwise, PCA is fit to the identified unique fixed
    points. This subspace is 3-dimensional if the RNN state dimensionality
    is >= 3.
    2) Plots the PCA representation of the stable unique fixed points as
    black dots.
    3) Plots the PCA representation of the unstable unique fixed points as
    red dots.
    4) Plots the PCA representation of the modes of the Jacobian at each
    fixed point. By default, only unstable modes are plotted.
    5) (optional) Plots example RNN state trajectories as blue lines.
    Args:
        fps: a FixedPoints object. See FixedPoints.py.
        state_traj (optional): [n_batch x n_time x n_states] numpy
        array or LSTMStateTuple with .c and .h as
        [n_batch x n_time x n_states/2] numpy arrays. Contains example
        trials of RNN state trajectories.
        plot_batch_idx (optional): Indices specifying which trials in
        state_traj to plot on top of the fixed points. Default: plot all
        trials.
        plot_start_time (optional): int specifying the first timestep to
        plot in the example trials of state_traj. Default: 0.
        plot_stop_time (optional): int specifying the last timestep to
        plot in the example trials of stat_traj. Default: n_time.
        stop_time (optional):
        mode_scale (optional): Non-negative float specifying the scaling
        of the plotted eigenmodes. A value of 1.0 results in each mode
        plotted as a set of diametrically opposed line segments
        originating at a fixed point, with each segment's length specified
        by the magnitude of the corresponding eigenvalue.
        fig (optional): Plotly figure upon which to plot.
    Returns:
        None.
    '''

    if fig is None:
        fig = go.Figure()

    if state_traj is not None:

        state_traj_bxtxd = state_traj
        [n_batch, n_time, n_states] = state_traj_bxtxd.shape

        plot_start_time = np.max([plot_start_time, 0])

        if plot_stop_time is None:
            plot_stop_time = n_time
        else:
            plot_stop_time = np.min([plot_stop_time, n_time])

        plot_time_idx = list(range(plot_start_time, plot_stop_time))

    n_inits = fps.n
    n_states = fps.n_states

    if n_states >= 3:
        pca = PCA(n_components=3)

        if state_traj is not None:
            state_traj_btxd = np.reshape(state_traj_bxtxd,
                                         (n_batch * n_time, n_states))
            pca.fit(state_traj_btxd)
        else:
            pca.fit(fps.xstar)

    if state_traj is not None:
        if plot_batch_idx is None:
            plot_batch_idx = list(range(n_batch))

        for batch_idx in plot_batch_idx:
            x_idx = state_traj_bxtxd[batch_idx]
            if output_colors is not None:
                x_colors = torch.nn.functional.sigmoid(output_colors[batch_idx])

            if n_states >= 3:
                z_idx = pca.transform(x_idx[plot_time_idx, :])
            else:
                z_idx = x_idx[plot_time_idx, :]
            if output_colors is not None:
                for i in range(len(z_idx) - 1):
                    fig.add_trace(go.Scatter3d(x=z_idx[i:i + 2, 0], y=z_idx[i:i + 2, 1], z=z_idx[i:i + 2, 2],
                                               mode='lines',
                                               line=dict(
                                                   color=f'rgb({x_colors[plot_time_idx[i], 0] * 255},{x_colors[plot_time_idx[i], 1] * 255},{x_colors[plot_time_idx[i], 2] * 255})')))
            else:
                fig.add_trace(go.Scatter3d(x=z_idx[:, 0], y=z_idx[:, 1], z=z_idx[:, 2],
                                           mode='lines',
                                           line=dict(color='blue', width=2)))

    for init_idx in range(n_inits):
        plot_fixed_point_plotly(
            fig,
            fps[init_idx],
            pca,
            scale=mode_scale)

    fig.show()
    return fig, pca


def plot_fixed_point_plotly(fig, fp, pca,
                            scale=1.0,
                            max_n_modes=3,
                            do_plot_unstable_fps=True,
                            do_plot_stable_modes=False,
                            stable_color='black',
                            stable_marker='circle',
                            unstable_color='red',
                            unstable_marker='cross',
                            **kwargs):
    '''Plots a single fixed point and its dominant eigenmodes.
    Args:
        fig: Plotly figure on which to plot everything.
        fp: a FixedPoints object containing a single fixed point
        (i.e., fp.n == 1),
        pca: PCA object as returned by sklearn.decomposition.PCA. This
        is used to transform the high-d state space representations
        into 3-d for visualization.
        scale (optional): Scale factor for stretching (>1) or shrinking
        (<1) lines representing eigenmodes of the Jacobian. Default:
        1.0 (unity).
        max_n_modes (optional): Maximum number of eigenmodes to plot.
        Default: 3.
        do_plot_stable_modes (optional): bool indicating whether or
        not to plot lines representing stable modes (i.e.,
        eigenvectors of the Jacobian whose eigenvalue magnitude is
        less than one).
    Returns:
        None.
    '''

    xstar = fp.xstar
    J = fp.J_xstar
    n_states = fp.n_states

    has_J = J is not None

    if has_J:

        if not fp.has_decomposed_jacobians:
            fp.decompose_Jacobians()

        e_vals = fp.eigval_J_xstar[0]
        e_vecs = fp.eigvec_J_xstar[0]

        sorted_e_val_idx = np.argsort(np.abs(e_vals))

        if max_n_modes > n_states:
            max_n_modes = n_states

        is_stable = np.all(np.abs(e_vals) < 1.0)

        if is_stable:
            color = stable_color
            marker = stable_marker
        else:
            color = unstable_color
            marker = unstable_marker
    else:
        color = stable_color
        marker = stable_marker

    do_plot = (not has_J) or is_stable or do_plot_unstable_fps

    if do_plot:
        if has_J:
            for mode_idx in range(max_n_modes):
                idx = sorted_e_val_idx[-(mode_idx + 1)]

                e_val_mag = np.abs(e_vals[idx])

                if e_val_mag > 1.0 or do_plot_stable_modes:

                    e_vec = np.real(e_vecs[:, idx])

                    xstar_plus = xstar + scale * e_val_mag * e_vec
                    xstar_minus = xstar - scale * e_val_mag * e_vec

                    xstar_mode = np.vstack((xstar_minus, xstar, xstar_plus))

                    if e_val_mag < 1.0:
                        color = stable_color
                    else:
                        color = unstable_color

                    if n_states >= 3 and pca is not None:
                        zstar_mode = pca.transform(xstar_mode)
                    else:
                        zstar_mode = xstar_mode

                    plot_123d_plotly(fig, zstar_mode,
                                     **kwargs)

        if n_states >= 3 and pca is not None:
            zstar = pca.transform(xstar)
        else:
            zstar = xstar

        fig.add_trace(go.Scatter3d(x=zstar[:, 0], y=zstar[:, 1], z=zstar[:, 2],
                                   mode='markers',
                                   marker=dict(color=color, symbol=marker, size=5),
                                   **kwargs))


def plot_123d_plotly(fig, z, **kwargs):
    '''Plots in 1D, 2D, or 3D.
    Args:
        fig: Plotly figure on which to plot everything.
        z: [n x n_states] numpy array containing data to be plotted,
        where n_states is 1, 2, or 3.
        any keyword arguments that can be passed to fig.add_trace(...).
    Returns:
        None.
    '''
    n_states = z.shape[1]
    if n_states == 3:
        fig.add_trace(go.Scatter3d(x=z[:, 0], y=z[:, 1], z=z[:, 2], **kwargs))
    elif n_states == 2:
        fig.add_trace(go.Scatter(x=z[:, 0], y=z[:, 1], **kwargs))
    elif n_states == 1:
        fig.add_trace(go.Scatter(x=np.arange(len(z)), y=z[:, 0], **kwargs))
