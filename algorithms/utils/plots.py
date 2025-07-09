import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

import wandb


def plot_all(values, conf, labels, xlabel, ylabel, title, iteration):
    for j, env in enumerate(values.keys()):
        fig, ax = plt.subplots(1, 1, figsize=set_size())

        x_mult = conf[f"{env}"]["NUM_STEPS"] * conf[f"{env}"]["NUM_ENVS"]

        legend = []
        for i, value in enumerate(values[env]):
            val = values[env][i]
            val_df = pd.DataFrame({f"vals_{i}": val[i] for i in range(len(val))})

            val_ewm = val_df.ewm(span=200, axis=0).mean().to_numpy().T

            mean = val_ewm.mean(0)

            xs = jnp.arange(len(mean)) * x_mult

            std = jnp.std(val_ewm, axis=0) / jnp.sqrt(val_ewm.shape[0])

            results_max = mean + std
            results_min = mean - std

            (leg,) = ax.plot(xs, mean, label=labels[i], linewidth=0.4)
            legend.append(leg)
            ax.fill_between(x=xs, y1=results_min, y2=results_max, alpha=0.5)

        if j == 0:
            ax.legend(
                legend,
                labels,
                loc="lower right",
                ncols=1,
            )

        if len(values.keys()) > 1:
            ax.set_title(env, fontsize=8)
            ax.set_xlabel(xlabel)
            ax.tick_params(axis="x", which="major", pad=-3)
            ax.tick_params(axis="y", which="major", pad=-3)

        else:
            ax.set_title(env, fontsize=8)
            ax.set_xlabel(xlabel)

        ax.set_ylabel(ylabel)

        os.makedirs(f"save_files/return_curves/{env}/", exist_ok=True)

        fig.savefig(
            f"save_files/return_curves/{env}/{title}_{ylabel}_{env}.pdf",
            format="pdf",
            bbox_inches="tight",
        )

        fig.savefig(
            f"save_files/return_curves/{env}/{title}_{ylabel}_{env}.png",
            format="png",
            bbox_inches="tight",
        )
        wandb.log(
            {
                f"{env}/return_curve_{title}": wandb.Image(
                    f"save_files/return_curves/{env}/{title}_{ylabel}_{env}.png"
                )
            },
            step=iteration,
        )


def set_size(width=487.8225, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    golden_ratio = 1.2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    subplots = (1, 1)
    fig_height_in = fig_width_in / golden_ratio * (subplots[0] / subplots[1])
    # Figure height in inches
    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
