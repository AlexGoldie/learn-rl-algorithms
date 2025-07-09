import argparse
import os
import os.path as osp
import warnings
from datetime import datetime
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from evosax import ParameterReshaper
from jax import lax

import wandb
from algorithms.evaluation.eval_open_recurrent_bb import eval_func
from algorithms.utils.learned_optimization.learned_optimization.learned_optimizers import (
    common,
)

api = wandb.Api()

warnings.simplefilter(action="ignore", category=FutureWarning)


def _second_moment_normalizer(x, axis, eps=1e-5):
    return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def transform_data_OPEN(p, g, mom, dorm, layer_prop, train_prop, batch_prop):
    eps1 = 1e-13
    p = jnp.expand_dims(p, -1)
    g = jnp.expand_dims(g, -1)

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
    # once normalized, add features that are constant across tensor.
    # namly the training step embedding.

    inp = jnp.expand_dims(train_prop, -1)

    stacked_batch_prop = jnp.expand_dims(batch_prop, -1)

    layer_prop = layer_prop

    stacked_layer_prop = layer_prop

    inp = jnp.concatenate([inp, stacked_layer_prop], axis=-1)

    inp = jnp.concatenate([inp, stacked_batch_prop], axis=-1)

    batch_dorm = dorm

    inp = jnp.concatenate([inp, batch_dorm], axis=-1)

    inp_g = jnp.concatenate([inp_stack_g, inp], axis=-1)

    return inp_g


@flax.struct.dataclass
class TrainState:
    params: Any
    rolling_features: common.MomAccumulator
    iteration: jnp.ndarray
    carry: Any
    rng: jnp.ndarray


class Distilled_OPEN:
    def __init__(
        self,
        exp_mult=0.001,
        step_mult=0.001,
        hidden_size=32,
        gru_features=16,
    ):
        super().__init__()
        self._step_mult = step_mult
        self._exp_mult = exp_mult

        self.gru = nn.GRUCell(features=gru_features)
        self.gru_features = gru_features

        self._mod = nn.Sequential(
            [
                nn.Dense(hidden_size),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(hidden_size),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(3),
            ]
        )

    def init(self, key, pholder):
        key = jax.random.split(key, 5)

        proxy_carry = self.gru.initialize_carry(key[4], (1,))

        return {
            "params": self._mod.init(key[0], jnp.zeros([self.gru_features])),
            "gru_params": self.gru.init(key[2], proxy_carry, pholder),
        }


def get_data_OPEN(key, points, seq_length):
    num_points = points
    train_len = 2000
    rng = jax.random.split(key, 5)
    training_step = jax.random.randint(
        rng[0], [num_points], minval=0, maxval=train_len - seq_length - 1
    )

    decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])

    p_init = jax.random.normal(rng[1], shape=(points,)) * config["p_std"]

    layer_props = jax.random.choice(
        rng[2], jnp.array([0.0, 0.5, 1.0]), replace=True, shape=(num_points, 1)
    )
    carry = meta_network.gru.initialize_carry(rng[3], (points, 1))

    moms = common.vec_rolling_mom(decays).init(p_init)

    train_state = TrainState(
        params=p_init,
        rolling_features=moms,
        iteration=training_step,
        carry=carry,
        rng=rng[4],
    )

    def single_step(train_state, xs=None):
        curr_params = train_state.params
        curr_mom = train_state.rolling_features
        iterate = train_state.iteration
        curr_carry = train_state.carry
        rng = train_state.rng

        train_prop = (iterate) / (train_len - 1)
        batch_prop = (iterate % 8) / (8 - 1)
        rng, rng_ = jax.random.split(rng)

        iterate += 1
        dorm = jnp.clip(
            jax.random.normal(rng_, shape=(points, 1)) * config["dorm_std"] + 1,
            0,
            10,
        )
        rng, rng_ = jax.random.split(rng)
        g = jax.random.normal(rng_, shape=(points,)) * config["g_std"]
        new_mom = common.vec_rolling_mom(decays).update(curr_mom, g)
        rng, rng_ = jax.random.split(rng)

        rand = jax.random.normal(rng_, (num_points,))
        inp = transform_data_OPEN(
            curr_params, g, new_mom.m, dorm, layer_props, train_prop, batch_prop
        )

        new_carry, gru_output = meta_network.gru.apply(
            meta_params["gru_params"], curr_carry, inp
        )
        output = meta_network._mod.apply(meta_params["params"], gru_output)

        new_updates = (
            output[..., 0] * 1e-3 * jnp.exp(output[..., 1] * 1e-3)
            + output[..., 2] * 1e-3 * rand
        )
        new_updates = jax.tree_util.tree_map(
            lambda x, mu: x - mu, new_updates, new_updates.mean()
        )
        new_p = curr_params - new_updates

        new_train_state = TrainState(
            params=new_p,
            rolling_features=new_mom,
            iteration=iterate,
            carry=new_carry,
            rng=rng,
        )

        return new_train_state, (
            inp,
            rand,
            (
                curr_params,
                g,
                new_mom.m,
                dorm,
                layer_props,
                train_prop,
                batch_prop,
                rand,
            ),
        )

    train_state, inputs = jax.lax.scan(
        jax.jit(single_step), train_state, xs=None, length=seq_length
    )

    return inputs, train_state.params


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
    parser.add_argument("--seq-length", type=int, default=100)
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

    wandb.init(
        project="meta-analysis",
        config=config,
        name=f"OPEN_recurrent_distil",
    )

    assert args.exp_name and args.exp_num or args.file_name

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

    meta_network = Distilled_OPEN(
        hidden_size=args.base_hsize, gru_features=int(args.base_hsize / 2)
    )
    pholder = meta_network.init(jax.random.PRNGKey(0), jnp.zeros((2, 19)))

    param_reshaper = ParameterReshaper(pholder)
    meta_params = param_reshaper.reshape_single(params)

    rng = jax.random.PRNGKey(100)

    get_data = get_data_OPEN
    pholder_in = jnp.zeros((2, 19))

    model = Distilled_OPEN(hidden_size=args.hsize, gru_features=int(args.hsize / 2))

    rng, rng_ = jax.random.split(rng)

    params = model.init(rng_, pholder_in)

    rng, key = jax.random.split(rng)

    if jax.local_device_count() > 1:
        pmap = True
    else:
        pmap = False

    epochs, iters = args.epochs, args.iters
    num_batches = args.num_batches
    lr = optax.linear_schedule(args.lr, args.lr_stop, epochs * iters * num_batches)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    rng = jax.random.PRNGKey(0)

    test_inp, test_targ = get_data(
        jax.random.PRNGKey(50), 10000, seq_length=args.seq_length
    )

    step = 0

    for i in range(epochs):
        step += 1

        def unroll_gru(dist_params, inputs, initial_state):
            """
            Unrolls the GRU over the sequence externally using jax.lax.scan.

            Args:
                params: Parameters of the GRU model.
                inputs: Input sequences with shape [batch_size, seq_length, feature_dim].
                initial_state: Initial hidden state for the GRU (carry) with shape [batch_size, hidden_size].

            Returns:
                outputs: GRU outputs for each time step, shape [batch_size, seq_length, output_dim].
                final_state: Final hidden state after processing the sequence.
            """

            def step_fn(state, inp):
                # _, rand, full_inputs = inp
                _, g, new_mom, dorm, layer_props, train_prop, batch_prop, rand = inp

                carry = state[0]
                params = state[1]

                inp = transform_data_OPEN(
                    params, g, new_mom, dorm, layer_props, train_prop, batch_prop
                )

                new_carry, gru_output = model.gru.apply(
                    dist_params["gru_params"], carry, inp
                )
                output = model._mod.apply(dist_params["params"], gru_output)

                new_updates = (
                    output[..., 0] * 1e-3 * jnp.exp(output[..., 1] * 1e-3)
                    + output[..., 2] * 1e-3 * rand
                )
                new_updates = jax.tree_util.tree_map(
                    lambda x, mu: x - mu, new_updates, new_updates.mean()
                )
                new_p = params - new_updates

                new_state = (new_carry, new_p)

                return new_state, new_p

            inputs = inputs[2]
            final_state, outputs = jax.lax.scan(step_fn, initial_state, inputs)
            # Convert outputs back to batch-major order: [batch_size, seq_length, output_dim]
            return final_state[1]

        def train_recurrent(
            params,
            opt_state,
            seq_length,
            batch_size,
            num_batches,
            iters,
            test_data,
            test_targ,
            key,
        ):
            # Define the loss function that unrolls the GRU externally.
            def loss_fn(params, inputs, targets, initial_state):
                # Unroll the GRU over the sequence.
                # inputs = jax.tree_util.tree_map(lambda x: jax.numpy.swapaxes(x, 0,1), inputs)
                outputs = unroll_gru(params, inputs, initial_state)
                return jnp.mean((outputs - targets) ** 2)

            @jax.jit
            def train_step(params, opt_state, inputs, targets, initial_state):
                loss, grads = jax.value_and_grad(loss_fn)(
                    params, inputs, targets, initial_state
                )
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss

            for epoch in range(iters):
                epoch_loss = 0.0

                for j in range(num_batches):
                    key, subkey = jax.random.split(key)
                    # get_data should now generate sequence data.
                    # inputs: [batch_size, seq_length, feature_dim]
                    # targets: [batch_size, seq_length, output_dim]
                    train_data, train_targets = get_data(subkey, batch_size, seq_length)

                    # Initialize the GRU hidden state (carry) for the batch.
                    initial_state = (
                        model.gru.initialize_carry(
                            jax.random.PRNGKey(0), (batch_size, 1)
                        ),
                        train_data[2][0][0, ...],
                    )

                    # Run one training step.
                    params, opt_state, batch_loss = train_step(
                        params, opt_state, train_data, train_targets, initial_state
                    )
                    epoch_loss += batch_loss

                epoch_loss /= num_batches

                # Evaluate on the test set.
                test_initial_state = (
                    model.gru.initialize_carry(jax.random.PRNGKey(0), (10000, 1)),
                    test_data[2][0][0, ...],
                )
                test_loss = loss_fn(params, test_data, test_targ, test_initial_state)

                print(
                    f"Epoch {epoch}, Train Loss: {epoch_loss:.6e}, Test Loss: {test_loss:.6e}"
                )
            wandb.log({"train_loss": epoch_loss, "test_loss": test_loss}, step=i)

            return params, opt_state

        params, opt_state = train_recurrent(
            params=params,
            opt_state=opt_state,
            seq_length=args.seq_length,
            batch_size=2048,
            num_batches=num_batches,
            iters=args.iters,
            test_data=test_inp,
            test_targ=test_targ,
            key=key,
        )

        returns = eval_func(
            envs=config["envs"],
            meta_params=params,
            iteration=i,
            title="",
            hsize=args.hsize,
        )

        save_loc = "save_files/open_recurrent_distil"
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
