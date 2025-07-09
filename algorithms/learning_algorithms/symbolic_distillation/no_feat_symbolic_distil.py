import argparse
import os
import warnings

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from evosax import ParameterReshaper
from jax import lax
from pysr import PySRRegressor

import wandb
from algorithms.evaluation.eval_no_feat_symbolic import eval_func
from algorithms.learning_algorithms.symbolic_distillation.utils import (
    create_jax_function,
)

api = wandb.Api()


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def _second_moment_normalizer(x, axis, eps=1e-5):
    return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def transform_data_no_feat(p, g, mom):
    if not p.shape:
        p = jnp.expand_dims(p, 0)

        # use gradient conditioning (L2O4RL)
        g = jnp.expand_dims(g, 0)

        mom = jnp.expand_dims(mom, 0)
        did_reshape = True
    else:
        did_reshape = False

    inps = []
    inp_g = []

    # batch_g = jnp.expand_dims(g, axis=-1)

    # **NO CURRENT GRADIENT**
    # feature consisting of raw parameter values
    # batch_p = jnp.expand_dims(p, axis=-1)
    inps.append(p)

    # feature consisting of all momentum values
    inps.append(mom)

    inp_stack = jnp.concatenate(inps, axis=-1)

    axis = list(range(len(p.shape)))

    inp_stack_g = jnp.concatenate([inp_stack, g], axis=-1)

    inp_stack_g = _second_moment_normalizer(inp_stack_g, axis=axis)

    return inp_stack_g


def get_data_no_feat(key, points):
    num_points = points
    rng_ = jax.random.split(key, 3)
    g = jax.random.normal(rng_[0], shape=(points, 1)) * config["g_std"]

    mom = jax.random.normal(rng_[1], shape=(points, 6)) * config["mom_std"]
    param = jax.random.normal(rng_[2], shape=(points, 1)) * config["p_std"]

    inp = transform_data_no_feat(param, g, mom)

    output = meta_network.apply(meta_params, inp)

    target = output[..., 0] * 1e-3 * jnp.exp(output[..., 1] * 1e-3)

    return inp, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", required=False, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--exp-num", type=int, default=None)
    parser.add_argument("--file-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--base-hsize", type=int, default=16)

    args = parser.parse_args()

    if args.envs == None:
        envs = ["breakout", "cartpole", "asterix", "ant", "spaceinvaders", "freeway"]
    else:
        envs = args.envs

    config = {
        "envs": envs,
        "target": "no_feat",
        "exp_name": args.exp_name,
        "exp_num": args.exp_num,
        "dorm_std": 1,
        "g_std": 0.5,
        "mom_std": 0.5,
        "p_std": 1,
    }

    wandb.init(
        project="meta-analysis",
        name="no_feat_symbolic",
        config=config,
    )

    assert args.file_name or (args.exp_name and args.exp_num)

    if args.exp_name:
        run_path = f"meta-analysis/{args.exp_name}"
        run = api.run(run_path)
        restored = wandb.restore(
            f"curr_param_{args.exp_num}.npy",
            run_path=run_path,
            root="save_files/params/",
            replace=True,
        )

        params = jnp.array(jnp.load(restored.name, allow_pickle=True))

    else:
        params = jnp.array(jnp.load(args.file_name, allow_pickle=True))

    meta_network = nn.Sequential(
        [
            nn.Dense(args.base_hsize),
            nn.LayerNorm(),
            nn.relu,
            nn.Dense(args.base_hsize),
            nn.LayerNorm(),
            nn.relu,
            nn.Dense(2),
        ]
    )

    test_case = jnp.zeros([2, 8])
    pholder = meta_network.init(jax.random.PRNGKey(0), test_case)
    param_reshaper = ParameterReshaper(pholder)

    meta_params = param_reshaper.reshape_single(params)

    rng = jax.random.PRNGKey(100)

    if jax.local_device_count() > 1:
        pmap = True
    else:
        pmap = False

    get_data = get_data_no_feat

    model = PySRRegressor(
        maxsize=60,
        populations=160,
        turbo=True,
        niterations=10,  # < Increase me for better results
        binary_operators=["+", "*", "-", "max", "min", "/"],
        unary_operators=[
            "exp",
            "neg",
            "tanh",
            "abs",
            "relu",
        ],
        elementwise_loss="L2DistLoss()",
        warm_start=True,
        print_precision=3,
        batching=True,
        batch_size=5000,
        maxdepth=9,
        weight_optimize=0.001,
    )

    epochs = args.epochs

    rng = jax.random.PRNGKey(0)
    rng, rng_ = jax.random.split(rng)
    train_inp, train_targ = get_data(rng_, 1000000)
    test_inp, test_targ = get_data(jax.random.PRNGKey(50), 10000)
    inp = train_inp
    step = 0
    for i in range(epochs):
        step += 1

        model.fit(inp, train_targ)

        best = model.equations_.iloc[-1]

        best_eq = best.equation
        print(best_eq)

        # Create building table!
        x = wandb.Table(
            dataframe=model.equations_[["complexity", "loss", "score", "equation"]]
        )
        wandb.log({f"tables/{i}": x}, step=i)

        jax_func = create_jax_function(best_eq)

        wandb.log({"current_fitness": best.loss}, step=i)
        y = jax_func(test_case)

        if isinstance(y, jnp.ndarray):
            if len(y.shape) > 0:
                wandb.log({"current_fitness": best.loss}, step=i)

                returns = eval_func(
                    config["envs"],
                    jax_func,
                    iteration=i,
                    title="best_loss",
                )
            else:
                print("function just a scalar!")
        else:
            print("function not an array!")

        # Best_in_PYSR
        jax_func = create_jax_function(model.get_best().equation)
        y = jax_func(test_case)

        if isinstance(y, jnp.ndarray):
            if len(y.shape) > 0:
                returns = eval_func(
                    config["envs"],
                    jax_func,
                    iteration=i,
                    title="best_score",
                )
            else:
                print("function just a scalar!")
        else:
            print("function not an array!")

        plt.close("all")
