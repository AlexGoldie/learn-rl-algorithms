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

import wandb
from algorithms.evaluation.eval_lpo_bb import eval_func

api = wandb.Api()


warnings.simplefilter(action="ignore", category=FutureWarning)


class Distilled_LPO(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size, use_bias=False)(x)
        x = nn.tanh(x)
        x = nn.Dense(1, use_bias=False)(x)

        return x

    def make_drift(self, x, params_drift):
        ratio, advantage = x[0], x[1]
        eps = 0.2

        def ppo_drift(ratio, advantage):
            return nn.relu(
                (ratio - jnp.clip(ratio, a_min=1 - eps, a_max=1 + eps)) * advantage
            )

        def drift_fn(ratio, advantage):
            log_ratio = jnp.log(ratio)
            r1 = ratio - 1
            meta_inputs = jnp.stack(
                [r1, r1 * r1, log_ratio, log_ratio * log_ratio], axis=-1
            )
            meta_inputs = jnp.concatenate(
                [meta_inputs, meta_inputs * advantage[..., None]], axis=-1
            )
            out = self.apply(params_drift, meta_inputs).squeeze(-1)
            drift_ppo = ppo_drift(ratio, advantage)
            if config.get("REMOVE_PPO"):
                print("NO PPO")
                drift_ppo = jnp.zeros_like(drift_ppo)
            drift = nn.relu(drift_ppo + out - 1e-6)
            return drift

        def drift_fn_sum(ratio, advantage):
            return drift_fn(ratio, advantage).sum()

        def policy_loss(ratio, advantage):
            drift = drift_fn(ratio, advantage)
            return -(ratio * advantage - drift).mean()

        policy_loss_grad = jax.grad(policy_loss)

        return policy_loss_grad(ratio, advantage)


def get_data_LPO():
    advantages = jnp.arange(-512, 512) / (64)
    ratios = jnp.exp(jnp.arange(-512, 512) / (256))
    ratio_and_advantage = jnp.transpose(
        jnp.array(
            [jnp.tile(ratios, len(advantages)), jnp.repeat(advantages, len(ratios))]
        )
    )

    ratio = ratio_and_advantage[:, 0]
    advantage = ratio_and_advantage[:, 1]

    eps = 0.2

    def ppo_drift(ratio, advantage):
        return nn.relu(
            (ratio - jnp.clip(ratio, a_min=1 - eps, a_max=1 + eps)) * advantage
        )

    def make_drift_fn_network(drift_apply, params_drift):
        def drift_fn(ratio, advantage):
            log_ratio = jnp.log(ratio)
            r1 = ratio - 1
            meta_inputs = jnp.stack(
                [r1, r1 * r1, log_ratio, log_ratio * log_ratio], axis=-1
            )
            meta_inputs = jnp.concatenate(
                [meta_inputs, meta_inputs * advantage[..., None]], axis=-1
            )
            out = drift_apply(params_drift, meta_inputs).squeeze(-1)
            drift_ppo = ppo_drift(ratio, advantage)
            if config.get("REMOVE_PPO"):
                print("NO PPO")
                drift_ppo = jnp.zeros_like(drift_ppo)
            drift = nn.relu(drift_ppo + out - 1e-6)
            return drift

        def drift_fn_sum(ratio, advantage):
            return drift_fn(ratio, advantage).sum()

        def policy_loss(ratio, advantage):
            drift = drift_fn(ratio, advantage)
            return -(ratio * advantage - drift).mean()

        drift_fn_grad = jax.grad(drift_fn_sum)
        policy_loss_grad = jax.grad(policy_loss)

        return drift_fn, drift_fn_grad, policy_loss_grad

    drift_fn, drift_fn_grad, policy_loss_grad = make_drift_fn_network(
        meta_network.apply, meta_params
    )

    target = policy_loss_grad(ratio, advantage)

    return (ratio, advantage), target


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
    parser.add_argument(
        "--hsize", type=int, default=128
    )  # Control the size of the distilled network
    parser.add_argument(
        "--base-hsize", type=int, default=128
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
    }

    assert args.file_name or (args.exp_name and args.exp_num)

    wandb.init(
        project="meta-analysis",
        config=config,
        name=f"LPO_distil",
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

    meta_network = Distilled_LPO(args.base_hsize)
    pholder = meta_network.init(jax.random.PRNGKey(0), jnp.zeros((2, 8)))
    param_reshaper = ParameterReshaper(pholder)
    meta_params = param_reshaper.reshape_single(params)

    rng = jax.random.PRNGKey(100)

    pholder_in = jnp.zeros((2, 8))
    get_data = get_data_LPO

    model = Distilled_LPO(args.hsize)

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

    step = 0
    for i in range(epochs):
        step += 1

        def train(params, opt_state):
            key = jax.random.PRNGKey(42)

            @jax.jit
            def loss_fn(params, inputs, targets):
                preds = model.make_drift(inputs, params)

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
                    train_data, train_targ = get_data_LPO()

                    params, opt_state, batch_loss = jax.jit(train_step)(
                        params, opt_state, train_data, train_targ
                    )
                    epoch_loss += batch_loss

                epoch_loss /= num_batches

                if epoch % 5 == 0:
                    print(f"Epoch {epoch}, Train Loss: {epoch_loss:.6e}")

            wandb.log({"train_loss": epoch_loss}, step=i)

            return params, opt_state

        params, opt_state = train(params, opt_state)

        returns = eval_func(
            envs=config["envs"],
            iteration=i,
            meta_params=params,
            hsize=args.hsize,
        )

        save_loc = "save_files/lpo_distil"
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
