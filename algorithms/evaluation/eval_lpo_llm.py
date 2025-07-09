import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse
import json
import os
import os.path as osp
import time
from datetime import datetime
from typing import Any, NamedTuple, Sequence

import distrax
import flax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax.linen.initializers import constant, orthogonal
from optax import adam

import wandb
from algorithms.utils.configs import all_configs
from algorithms.utils.plots import plot_all
from algorithms.utils.wrappers import (
    AutoResetEnvWrapper,
    BraxGymnaxWrapper,
    ClipAction,
    FlatWrapper,
    GymnaxGymWrapper,
    GymnaxLogWrapper,
    NormalizeObservation,
    NormalizeReward,
    TransformObservation,
    VecEnv,
)

api = wandb.Api()


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: dict

    @nn.compact
    def __call__(self, x):
        hsize = self.config["HSIZE"]
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        if self.config["CONTINUOUS"]:
            actor_logtstd = self.param(
                "log_std", nn.initializers.zeros, (self.action_dim,)
            )
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


@flax.struct.dataclass
class OptState:
    params: Any
    state: Any
    iteration: jnp.ndarray
    adam_state: Any = (None,)


def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["TOTAL_UPDATES"] = (
        config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
    )

    def train(disco_drift, rng):

        if "Brax-" in config["ENV_NAME"]:
            name = config["ENV_NAME"].split("Brax-")[1]

            env, env_params = BraxGymnaxWrapper(env_name=name), None
            if config.get("CLIP_ACTION"):
                env = ClipAction(env)
            env = GymnaxLogWrapper(env)

            if config.get("SYMLOG_OBS"):
                env = TransformObservation(env, transform_obs=symlog)

            env = VecEnv(env)
            if config.get("NORMALIZE"):
                env = NormalizeObservation(env)
                env = NormalizeReward(env, config["GAMMA"])
            network = ActorCritic(env.action_space(env_params).shape[0], config=config)
            init_x = jnp.zeros(env.observation_space(env_params).shape)
        else:
            # INIT ENV
            env, env_params = gymnax.make(config["ENV_NAME"])

            env = GymnaxGymWrapper(env, env_params, config)
            env = FlatWrapper(env)
            env = GymnaxLogWrapper(env)
            env = VecEnv(env)
            network = ActorCritic(env.action_space, config=config)
            init_x = jnp.zeros(env.observation_space)

        # INIT NETWORK

        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, init_x)

        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return config["LR"] * frac

        if config["ANNEAL_LR"]:
            opt = adam(
                learning_rate=linear_schedule,
                eps=1e-5,
            )
        else:
            opt = adam(learning_rate=config["LR"], eps=1e-5)
        clip_opt = optax.clip_by_global_norm(config["MAX_GRAD_NORM"])
        adam_state = opt.init(network_params)

        train_state = OptState(
            params=network_params,
            state=None,
            adam_state=adam_state,
            iteration=jnp.asarray(0, dtype=jnp.int32),
        )

        # INIT ENV
        all_rng = jax.random.split(_rng, config["NUM_ENVS"] + 1)
        rng, _rng = all_rng[0], all_rng[1:]
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, rng = runner_state
                rng, _rng = jax.random.split(rng)
                # SELECT ACTION
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, done, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)
            last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        log_ratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(log_ratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        drift = nn.relu(disco_drift(ratio, gae, config["CLIP_EPS"]))
                        loss_actor = -(ratio * gae - drift).mean()

                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        approx_kl_old = -(log_prob - traj_batch.log_prob).mean()
                        approx_kl = (
                            (ratio - 1) - (log_prob - traj_batch.log_prob)
                        ).mean()
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            approx_kl,
                            approx_kl_old,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )

                    grads, _ = clip_opt.update(grads, None)
                    updates, new_adam_state = opt.update(grads, train_state.adam_state)

                    new_params = optax.apply_updates(train_state.params, updates)

                    train_state = OptState(
                        params=new_params,
                        state=None,
                        adam_state=new_adam_state,
                        iteration=train_state.iteration + 1,
                    )
                    # jax.debug.print('train_state: {train}', train=train_state)

                    adam_update = jax.flatten_util.ravel_pytree(updates)[0]

                    adam_update_mean = jnp.mean(jnp.array(adam_update))
                    abs_adam_update_mean = jnp.mean(jnp.abs(jnp.asarray(adam_update)))

                    param_flat = jax.flatten_util.ravel_pytree(new_params)[0]
                    param_mean = jnp.mean(jnp.array(param_flat))
                    abs_mean = jnp.mean(jnp.abs(jnp.array(param_flat)))

                    total_loss_updates = (
                        total_loss,
                        (adam_update_mean, abs_adam_update_mean, param_mean, abs_mean),
                    )

                    return train_state, total_loss_updates

                train_state, traj_batch, advantages, targets, rng = update_state

                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)

                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_update = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            metric = traj_batch.info

            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, last_done, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return runner_state, metric

    return train


def eval_func(envs, function, num_runs=8, iteration=0, title=""):
    returns_list = dict()
    runtimes = dict()
    norm_fit = []

    pmap = jax.local_device_count() > 1
    try:
        for i, env in enumerate(envs):
            returns_list.update({envs[i]: []})
            runtimes.update({envs[i]: []})

            rng = jax.random.PRNGKey(42)
            all_configs[f"{env}"]["VISUALISE"] = True

            start = time.time()
            rngs = jax.random.split(rng, num_runs)
            if pmap:
                rngs = jnp.reshape(rngs, (jax.local_device_count(), -1, 2))
            else:
                rngs = jnp.reshape(rngs, (-1, 2))

            asdf = jax.jit(
                jax.vmap(
                    make_train(all_configs[f"{env}"]),
                    in_axes=(None, 0),
                )
            )

            if pmap:
                asdf = jax.pmap(asdf)
            out, metrics = asdf(jax.tree_util.Partial(function), rngs)

            fitness = metrics["returned_episode_returns"][..., -1, -1, :].mean()
            end = time.time()
            print(f"runtime = {end - start}")
            runtimes[envs[i]].append(end - start)
            print(f"{envs[i]}      learned fitness: {fitness}")
            returns = (
                metrics["returned_episode_returns"]
                .mean(-1)
                .mean(-1)
                .reshape(num_runs, -1)
            )
            returns_list[env].append(returns[:])
            save_dir = f"save_files/eval/{str(datetime.now()).replace(' ', '_')}"
            os.makedirs(f"{save_dir}", exist_ok=True)
            jnp.save(
                osp.join(save_dir, f"returns_{env}.npy"),
                returns,
            )

            wandb.save(
                osp.join(save_dir, f"returns_{env}.npy"),
                base_path=save_dir,
            )

            norm_fit.append(fitness / all_configs[env]["PPO_TEMP"])

            wandb.log({f"{env}/return_{title}": fitness}, step=iteration)
            wandb.log({f"{env}/runtime_{title}": end - start}, step=iteration)
            wandb.log(
                {f"{env}/norm_fit": fitness / all_configs[env]["PPO_TEMP"]},
                step=iteration,
            )

    except Exception as error:
        print(error)
        return None, error

    plot_all(
        returns_list,
        all_configs,
        ["LPO_LLM"],
        xlabel="Frames",
        ylabel=f"Return",
        title=title,
        iteration=iteration,
    )

    return jnp.array(norm_fit).mean(), None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", required=False, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--exp-num", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=16)

    args = parser.parse_args()

    if args.envs == None:
        envs = ["breakout", "cartpole", "asterix", "ant", "spaceinvaders", "freeway"]
    else:
        envs = args.envs

    config = {
        "envs": envs,
        "exp_name": args.exp_name,
        "exp_num": args.exp_num,
    }

    assert args.exp_name and args.exp_num

    wandb.init(
        project="meta-analysis",
        config=config,
        name=f"eval_lpo_llm",
    )
    run_path = f"meta-analysis/{args.exp_name}"
    run = api.run(run_path)
    artifacts = run.logged_artifacts()

    # Find the artifact named "archive"
    artifact = None
    for art in artifacts:
        if f"archive" in art.name:
            artifact = art

    if artifact:
        print(f"Found artifact: {artifact.name}")

        # Download the artifact
        artifact_dir = artifact.download()
        # Load the JSON table file
        table_path = f"{artifact_dir}/archive.table.json"

        with open(table_path, "r") as f:
            table_data = json.load(f)
    df = pd.DataFrame(table_data["data"], columns=table_data["columns"])

    eqn = df.iloc[args.exp_num - 1]["code"]

    namespace = {}
    exec(eqn, globals(), namespace)
    names = list(namespace.keys())
    func = namespace[names[0]]
    eval_func(
        envs=envs,
        num_runs=args.num_runs,
        iteration=0,
        function=func,
    )
