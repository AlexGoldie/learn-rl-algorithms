import argparse
import os
import os.path as osp
import warnings
from datetime import datetime

import flax.linen as nn
import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from evosax import ParameterReshaper
from jax import lax

import wandb
from algorithms.evaluation.eval_open_ff_bb import eval_func

api = wandb.Api()


warnings.simplefilter(action="ignore", category=FutureWarning)


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
    else:
        gsign = jnp.sign(g)
        glog = jnp.log(jnp.abs(g) + eps1)

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


class Distilled_OPEN(nn.Module):
    hsize: int = 16

    def setup(self):
        hsize = self.hsize
        self.distil = nn.Sequential(
            [
                nn.Dense(hsize),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(hsize),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(3),
            ]
        )

    def __call__(self, x):
        x, rand = x[0], x[1]
        distilled = self.distil(x)
        distilled_out = distilled[..., 0] * 1e-3 * jnp.exp(distilled[..., 1] * 1e-3)
        distilled = distilled_out + 1e-3 * distilled[..., 2] * rand
        return distilled


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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-stop", type=float, default=1e-5)
    parser.add_argument("--file-name", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--exp-num", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument(
        "--hsize", type=int, default=16
    )  # Control the size of the distilled network
    parser.add_argument(
        "--base-hsize", type=int, default=16
    )  # size of the original network

    args = parser.parse_args()

    if args.envs == None:
        envs = ["breakout", "cartpole", "asterix", "ant", "spaceinvaders", "freeway"]
    else:
        envs = args.envs

    config = {
        "envs": envs,
        "lr": args.lr,
        "exp_name": args.exp_name,
        "exp_num": args.exp_num,
        "iters": args.iters,
        "hsize": args.hsize,
        "dorm_std": 1,
        "g_std": 0.5,
        "mom_std": 0.5,
        "p_std": 1,
    }

    assert args.file_name or (args.exp_name and args.exp_num)
    global hsize
    hsize = args.hsize

    wandb.init(
        project="meta-analysis",
        config=config,
        name=f"OPEN_distil",
    )

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
    pholder = meta_network.init(jax.random.PRNGKey(0), jnp.zeros((2, 19)))

    param_reshaper = ParameterReshaper(pholder)
    meta_params = param_reshaper.reshape_single(params)

    rng = jax.random.PRNGKey(100)

    get_data = get_data_OPEN
    pholder_in = jnp.zeros((2, 19))

    model = Distilled_OPEN(args.hsize)

    rng, rng_ = jax.random.split(rng)

    params = model.init(rng_, pholder_in)

    if jax.local_device_count() > 1:
        pmap = True
    else:
        pmap = False

    epochs, iters = args.epochs, args.iters
    num_batches = 100
    lr = optax.linear_schedule(args.lr, args.lr_stop, epochs * iters * num_batches)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    rng = jax.random.PRNGKey(0)

    test_inp, test_targ = get_data(jax.random.PRNGKey(50), 10000)
    step = 0
    for i in range(epochs):
        step += 1

        def train(test_data, test_targ, params, opt_state):
            key = jax.random.PRNGKey(42)

            @jax.jit
            def loss_fn(params, inputs, targets):

                preds = model.apply(params, inputs)

                return jnp.mean(jnp.square(preds.squeeze() - targets))

            @jax.jit
            def train_step(params, opt_state, inputs, train_targ):
                loss, grads = jax.value_and_grad(loss_fn)(params, inputs, train_targ)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss

            for epoch in range(iters):
                epoch_loss = 0.0

                for j in range(num_batches):
                    key, subkey = jax.random.split(key)
                    train_data, train_targ = get_data(subkey, 2048)

                    params, opt_state, batch_loss = jax.jit(train_step)(
                        params, opt_state, train_data, train_targ
                    )
                    epoch_loss += batch_loss

                epoch_loss /= num_batches
                test_loss = loss_fn(params, test_data, test_targ)

                if epoch % 1 == 0:
                    print(
                        f"Epoch {epoch}, Train Loss: {epoch_loss:.6e}, Test Loss: {test_loss:.6e}"
                    )

            wandb.log({"train_loss": epoch_loss, "test_loss": test_loss}, step=i)

            return params, opt_state

        params, opt_state = train(test_inp, test_targ, params, opt_state)

        returns = eval_func(
            envs=config["envs"],
            meta_params=params,
            iteration=i,
            title="",
            hsize=args.hsize,
        )

        save_loc = "save_files/open_distil"
        os.makedirs(save_loc, exist_ok=True)
        save_dir = f"{save_loc}/{str(datetime.now()).replace(' ', '_')}"
        os.mkdir(f"{save_dir}")
        jnp.save(
            osp.join(save_dir, f"curr_param_{i}.npy"),
            jax.flatten_util.ravel_pytree(params)[0],
        )

        wandb.save(
            osp.join(save_dir, f"curr_param_{i}.npy"),
            base_path=save_dir,
        )

        plt.close("all")
