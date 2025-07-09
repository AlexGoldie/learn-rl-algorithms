import argparse
import os
import os.path as osp
import time
from datetime import datetime
from typing import NamedTuple, Sequence

import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from evosax import OpenES, ParameterReshaper
from flax.linen.initializers import constant, orthogonal

import wandb
from algorithms.utils.architectures.open_recurrent import recurrent_open_gru as optim
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


class Actor(nn.Module):
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
        actor_mean_activation_1 = {
            "kernel": jnp.mean(jnp.abs(actor_mean), axis=0),
            "bias": jnp.mean(jnp.abs(actor_mean), axis=0),
        }

        actor_mean = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean_activation_2 = {
            "kernel": jnp.mean(jnp.abs(actor_mean), axis=0),
            "bias": jnp.mean(jnp.abs(actor_mean), axis=0),
        }
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean_activation_3 = {
            "kernel": jnp.mean(jnp.abs(actor_mean), axis=0),
            "bias": jnp.mean(jnp.abs(actor_mean), axis=0),
        }

        if self.config["CONTINUOUS"]:
            actor_logtstd = self.param(
                "log_std", nn.initializers.zeros, (self.action_dim,)
            )
            actor_mean_activation_4 = jnp.expand_dims(
                jnp.mean(jnp.exp(actor_logtstd), axis=0), axis=0
            )
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        if self.config["CONTINUOUS"]:
            activations = (
                actor_mean_activation_1,
                actor_mean_activation_2,
                actor_mean_activation_3,
                actor_mean_activation_4,
            )
        else:
            activations = (
                actor_mean_activation_1,
                actor_mean_activation_2,
                actor_mean_activation_3,
            )

        return pi, activations


class Critic(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):
        hsize = self.config["HSIZE"]
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic_mean_activation_1 = {
            "kernel": jnp.mean(jnp.abs(critic), axis=0),
            "bias": jnp.mean(jnp.abs(critic), axis=0),
        }
        critic = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic_mean_activation_2 = {
            "kernel": jnp.mean(jnp.abs(critic), axis=0),
            "bias": jnp.mean(jnp.abs(critic), axis=0),
        }
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        critic_mean_activation_3 = {
            "kernel": jnp.mean(jnp.abs(critic), axis=0),
            "bias": jnp.mean(jnp.abs(critic), axis=0),
        }

        activations = (
            critic_mean_activation_1,
            critic_mean_activation_2,
            critic_mean_activation_3,
        )
        return jnp.squeeze(critic, axis=-1), activations


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


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

    def train(rng, meta_params):

        meta_opt = optim(
            hidden_size=config["OPTIM_HSIZE"],
            gru_features=int(config["OPTIM_HSIZE"] / 2),
        )
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
            actor = Actor(env.action_space(env_params).shape[0], config=config)
            critic = Critic(config=config)
            init_x = jnp.zeros(env.observation_space(env_params).shape)
        else:
            # INIT ENV
            env, env_params = gymnax.make(config["ENV_NAME"])
            env = GymnaxGymWrapper(env, env_params, config)
            env = FlatWrapper(env)

            env = GymnaxLogWrapper(env)
            env = VecEnv(env)
            actor = Actor(env.action_space, config=config)
            critic = Critic(config=config)
            init_x = jnp.zeros(env.observation_space)
        # INIT NETWORK

        rng, _rng = jax.random.split(rng)
        actor_params = actor.init(_rng, init_x)
        critic_params = critic.init(_rng, init_x)

        opt = meta_opt.opt_fn(meta_params)
        clip_opt = optax.clip_by_global_norm(config["MAX_GRAD_NORM"])

        rng, rng_act, rng_crit = jax.random.split(rng, 3)
        train_state_actor = opt.init(actor_params, key=rng_act)
        train_state_critic = opt.init(critic_params, key=rng_crit)

        act_param_tree = jax.tree_util.tree_structure(train_state_actor.params)
        act_layer_props = []
        num_act_layers = len(train_state_actor.params["params"])
        for i, layer in enumerate(train_state_actor.params["params"]):
            layer_prop = i / (num_act_layers - 1)
            if type(train_state_actor.params["params"][layer]) == dict:
                act_layer_props.extend(
                    [layer_prop] * len(train_state_actor.params["params"][layer])
                )
            else:
                act_layer_props.extend([layer_prop])

        act_layer_props = jax.tree_util.tree_unflatten(act_param_tree, act_layer_props)

        crit_param_tree = jax.tree_util.tree_structure(train_state_critic.params)
        crit_layer_props = []
        num_crit_layers = len(train_state_critic.params["params"])
        for i, layer in enumerate(train_state_critic.params["params"]):
            layer_prop = i / (num_crit_layers - 1)
            if type(train_state_critic.params["params"][layer]) == dict:
                crit_layer_props.extend(
                    [layer_prop] * len(train_state_critic.params["params"][layer])
                )
            else:
                crit_layer_props.extend([layer_prop])

        crit_layer_props = jax.tree_util.tree_unflatten(
            crit_param_tree, crit_layer_props
        )

        # INIT ENV
        all_rng = jax.random.split(_rng, config["NUM_ENVS"] + 1)
        rng, _rng = all_rng[0], all_rng[1:]
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state_actor,
                    train_state_critic,
                    env_state,
                    last_obs,
                    last_done,
                    rng,
                ) = runner_state
                rng, _rng = jax.random.split(rng)
                # SELECT ACTION
                pi, _ = actor.apply(train_state_actor.params, last_obs)
                value, _ = critic.apply(train_state_critic.params, last_obs)
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
                runner_state = (
                    train_state_actor,
                    train_state_critic,
                    env_state,
                    obsv,
                    done,
                    rng,
                )

                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state_actor,
                train_state_critic,
                env_state,
                last_obs,
                last_done,
                rng,
            ) = runner_state
            last_val, _ = critic.apply(train_state_critic.params, last_obs)
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
                def _update_minbatch(train_state_key, batch_info):
                    train_state_actor, train_state_critic, key = train_state_key
                    key, key_ = jax.random.split(key)

                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(actor_params, critic_params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, actor_activations = actor.apply(
                            actor_params, traj_batch.obs
                        )
                        value, critic_activations = critic.apply(
                            critic_params, traj_batch.obs
                        )

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
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        return total_loss, (actor_activations, critic_activations)

                    training_prop = (
                        train_state_actor.iteration
                        // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
                    ) / (config["NUM_UPDATES"] - 1)
                    batch_prop = (
                        (train_state_actor.iteration // config["NUM_MINIBATCHES"])
                        % config["UPDATE_EPOCHS"]
                    ) / (config["UPDATE_EPOCHS"] - 1)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, argnums=[0, 1])
                    (total_loss, (actor_activations, critic_activations)), (
                        actor_grads,
                        critic_grads,
                    ) = grad_fn(
                        train_state_actor.params,
                        train_state_critic.params,
                        traj_batch,
                        advantages,
                        targets,
                    )

                    key_actor, key_critic = jax.random.split(key_)
                    actor_grads, _ = clip_opt.update(actor_grads, None)
                    critic_grads, _ = clip_opt.update(critic_grads, None)
                    actor_mask = {"kernel": 1, "bias": 1}
                    critic_mask = {"kernel": 0, "bias": 0}

                    # FOR NOW, HARD CODE MASK
                    if config["CONTINUOUS"]:
                        mask = {
                            "actor": {
                                "params": {
                                    "Dense_0": actor_mask,
                                    "Dense_1": actor_mask,
                                    "Dense_2": actor_mask,
                                    "log_std": 1,
                                }
                            },
                            "critic": {
                                "params": {
                                    "Dense_0": critic_mask,
                                    "Dense_1": critic_mask,
                                    "Dense_2": critic_mask,
                                }
                            },
                        }

                    else:

                        mask = {
                            "actor": {
                                "params": {
                                    "Dense_0": actor_mask,
                                    "Dense_1": actor_mask,
                                    "Dense_2": actor_mask,
                                }
                            },
                            "critic": {
                                "params": {
                                    "Dense_0": critic_mask,
                                    "Dense_1": critic_mask,
                                    "Dense_2": critic_mask,
                                }
                            },
                        }

                    activations = {
                        "actor": actor_activations,
                        "critic": critic_activations,
                    }
                    grads = {"actor": actor_grads, "critic": critic_grads}
                    layer_props = {"actor": act_layer_props, "critic": crit_layer_props}

                    # APPLY OPTIMIZER
                    (
                        train_state_actor,
                        train_state_critic,
                        actor_updates,
                        critic_updates,
                    ) = opt.update(
                        train_state_actor,
                        train_state_critic,
                        grads,
                        activations,
                        key=key_actor,
                        training_prop=training_prop,
                        batch_prop=batch_prop,
                        layer_props=layer_props,
                        mask=mask,
                    )

                    train_state_key_ = (train_state_actor, train_state_critic, key)
                    return train_state_key_, (
                        total_loss,
                        actor_updates,
                        critic_updates,
                        actor_activations,
                        critic_activations,
                    )

                (
                    train_state_actor,
                    train_state_critic,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
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

                train_state_key = (train_state_actor, train_state_critic, rng)
                train_state_key, total_loss_updates = jax.lax.scan(
                    _update_minbatch, train_state_key, minibatches
                )
                train_state_actor, train_state_critic, rng = train_state_key
                update_state = (
                    train_state_actor,
                    train_state_critic,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )

                return update_state, total_loss_updates

            update_state = (
                train_state_actor,
                train_state_critic,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_update = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state_actor = update_state[0]
            train_state_critic = update_state[1]

            if config["VISUALISE"]:

                metric = traj_batch.info

            else:
                metric = dict()

                metric.update(
                    {
                        "returned_episode_returns": traj_batch.info[
                            "returned_episode_returns"
                        ][-1].mean(),
                    }
                )

            rng = update_state[-1]
            runner_state = (
                train_state_actor,
                train_state_critic,
                env_state,
                last_obs,
                last_done,
                rng,
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state_actor,
            train_state_critic,
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


def eval_func(envs, num_runs=8, iteration=0, title="", meta_params=None, hsize=16):

    pmap = jax.local_device_count() > 1
    returns_list = dict()
    runtimes = dict()

    meta_opt = optim(hidden_size=hsize, gru_features=int(hsize / 2))

    meta_params_pholder = meta_opt.init(jax.random.PRNGKey(0))

    strategy = OpenES(
        popsize=2,
        pholder_params=meta_params_pholder,
        opt_name="adam",
        centered_rank=True,
        maximize=True,
    )
    meta_params = jax.flatten_util.ravel_pytree(meta_params)[0]
    meta_params = jnp.tile(meta_params, (jax.local_device_count(), 1))
    meta_params = strategy.param_reshaper.reshape(meta_params)

    for i, env in enumerate(envs):

        returns_list.update({envs[i]: []})
        runtimes.update({envs[i]: []})

        rng = jax.random.PRNGKey(42)
        all_configs[f"{env}"]["VISUALISE"] = True
        all_configs[f"{env}"]["OPTIM_HSIZE"] = hsize

        start = time.time()
        rngs = jax.random.split(rng, num_runs)
        if pmap:
            rngs = jnp.reshape(rngs, (jax.local_device_count(), -1, 2))
        else:
            rngs = jnp.reshape(rngs, (-1, 2))

        asdf = jax.jit(
            jax.vmap(
                make_train(all_configs[f"{env}"]),
                in_axes=(0, None),
            )
        )

        asdf = jax.jit(
            jax.vmap(
                asdf,
                in_axes=(
                    None,
                    strategy.param_reshaper.vmap_dict,
                ),
            ),
        )

        if pmap:
            asdf = jax.pmap(asdf)

        out, metrics = asdf(rngs, meta_params)

        fitness = metrics["returned_episode_returns"][..., -1, -1, :].mean()
        end = time.time()
        print(f"runtime = {end - start}")
        runtimes[envs[i]].append(end - start)
        print(f"{envs[i]}      learned fitness: {fitness}")
        returns = (
            metrics["returned_episode_returns"].mean(-1).mean(-1).reshape(num_runs, -1)
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
        wandb.log({f"{env}/return_{title}": fitness}, step=iteration)
        wandb.log({f"{env}/runtime_{title}": end - start}, step=iteration)

    plot_all(
        returns_list,
        all_configs,
        ["OPEN_RECURRENT_BB"],
        xlabel="Frames",
        ylabel=f"Return",
        title=title,
        iteration=iteration,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", required=False, default=None)
    parser.add_argument("--file-name", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--exp-num", type=int, default=None)
    parser.add_argument("--hsize", type=int, default=128)
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
        "hsize": args.hsize,
    }

    assert args.file_name or (args.exp_name and args.exp_num)

    wandb.init(
        project="meta-analysis",
        config=config,
        name=f"eval_open_recurrent_bb",
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

    eval_func(
        envs=envs,
        meta_params=params,
        num_runs=args.num_runs,
        iteration=0,
        hsize=args.hsize,
    )
