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
from algorithms.evaluation.eval_open_symbolic import eval_func
from algorithms.learning_algorithms.symbolic_distillation.utils import (
    create_jax_function,
)

api = wandb.Api()


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def _second_moment_normalizer(x, axis, eps=1e-5):
    return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def transform_data_OPEN(p, g, mom, dorm, layer_prop, train_prop, batch_prop):
    eps1 = 1e-13

    if not p.shape:
        p = jnp.expand_dims(p, 0)

        # use gradient conditioning (L2O4RL)
        gsign = jnp.expand_dims(jnp.sign(g), 0)
        glog = jnp.expand_dims(jnp.log(jnp.abs(g) + eps1), 0)

        mom = jnp.expand_dims(mom, 0)
        did_reshape = True
    else:
        gsign = jnp.sign(g)
        glog = jnp.log(jnp.abs(g) + eps1)
        did_reshape = False

    inps = []
    inp_g = []

    batch_gsign = gsign
    batch_glog = glog

    batch_p = p
    inps.append(batch_p)

    # feature consisting of all momentum values
    momsign = jnp.sign(mom)
    momlog = jnp.log(jnp.abs(mom) + eps1)
    inps.append(momsign)
    inps.append(momlog)

    inp_stack = jnp.concatenate(inps, axis=-1)

    axis = list(range(len(p.shape)))

    inp_stack_g = jnp.concatenate([inp_stack, batch_gsign, batch_glog], axis=-1)
    inp_stack_g = _second_moment_normalizer(inp_stack_g, axis=axis)

    inp = train_prop

    stacked_batch_prop = batch_prop

    stacked_layer_prop = layer_prop

    inp = jnp.concatenate([inp, stacked_layer_prop], axis=-1)

    inp = jnp.concatenate([inp, stacked_batch_prop], axis=-1)

    batch_dorm = dorm

    inp = jnp.concatenate([inp, batch_dorm], axis=-1)

    inp_g = jnp.concatenate([inp_stack_g, inp], axis=-1)

    return inp_g


def get_data_OPEN(key, points):
    num_points = points
    rng_ = jax.random.split(key, 8)
    dorm = jnp.clip(
        jax.random.normal(rng_[0], shape=(points, 1)) * config["dorm_std"] + 1,
        0,
        10,
    )
    g = jax.random.normal(rng_[1], shape=(points, 1)) * config["g_std"]
    layer_props = jax.random.choice(
        rng_[2], jnp.array([0.0, 0.5, 1.0]), replace=True, shape=(num_points, 1)
    )

    mom = jax.random.normal(rng_[3], shape=(points, 6)) * config["mom_std"]
    param = jax.random.normal(rng_[4], shape=(points, 1)) * config["p_std"]
    train_prop = jax.random.uniform(rng_[5], shape=(num_points, 1), minval=0, maxval=1)
    batch_prop = jax.random.uniform(rng_[6], shape=(num_points, 1), minval=0, maxval=1)

    inp = transform_data_OPEN(param, g, mom, dorm, layer_props, train_prop, batch_prop)

    rand = jax.random.normal(rng_[7], (num_points,))

    output = meta_network.apply(meta_params, inp)

    target = (
        output[..., 0] * 1e-3 * jnp.exp(output[..., 1] * 1e-3)
        + output[..., 2] * 1e-3 * rand
    )

    return (inp, rand), target


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
            nn.Dense(3),
        ]
    )
    test_case = jnp.zeros((2, 19))

    pholder = meta_network.init(jax.random.PRNGKey(0), test_case)
    param_reshaper = ParameterReshaper(pholder)

    meta_params = param_reshaper.reshape_single(params)

    rng = jax.random.PRNGKey(100)

    if jax.local_device_count() > 1:
        pmap = True
    else:
        pmap = False

    get_data = get_data_OPEN

    model = PySRRegressor(
        maxsize=60,
        populations=160,
        turbo=True,
        niterations=10,  # < Increase me for better results
        binary_operators=["+", "*", "-", "/", "max", "min"],
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

    inp = jnp.concatenate([train_inp[0], jnp.expand_dims(train_inp[1], -1)], -1)
    test_inp = jnp.concatenate([test_inp[0], jnp.expand_dims(test_inp[1], -1)], -1)

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
