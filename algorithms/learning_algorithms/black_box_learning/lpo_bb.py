import argparse
import os
import os.path as osp
import time
from datetime import datetime
from functools import partial
from typing import Any, NamedTuple, Sequence

import distrax
import flax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from evosax import FitnessShaper, OpenES, ParameterReshaper
from evosax.utils import ESLog
from flax.linen.initializers import constant, orthogonal
from optax import adam
from tqdm import tqdm

import wandb
from algorithms.evaluation.eval_lpo_bb import eval_func
from algorithms.utils.architectures.drift import MetaDrift
from algorithms.utils.configs import all_configs
from algorithms.utils.wrappers import (
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
    meta_network = MetaDrift()

    @partial(jax.jit, static_argnames=["grid_type"])
    def train(meta_params, rng, grid_type=6):
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
            env, env_params = gymnax.make(config["ENV_NAME"])
            env = GymnaxGymWrapper(env, env_params, config)
            env = FlatWrapper(env)

            env = GymnaxLogWrapper(env)
            env = VecEnv(env)
            network = ActorCritic(env.action_space, config=config)
            init_x = jnp.zeros(env.observation_space)

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
                b1=config["B1"],
                b2=config["B2"],
            )
        else:
            opt = adam(
                learning_rate=config["LR"], eps=1e-5, b1=config["B1"], b2=config["B2"]
            )
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

                        r1 = ratio - 1
                        meta_inputs = jnp.stack(
                            [r1, r1 * r1, log_ratio, log_ratio * log_ratio], axis=-1
                        )
                        meta_inputs = jnp.concatenate(
                            [meta_inputs, meta_inputs * gae[..., None]], axis=-1
                        )
                        drift = meta_network.apply(meta_params, meta_inputs)
                        ppo_drift = (
                            ratio
                            - jnp.clip(
                                ratio,
                                a_min=1 - config["CLIP_EPS"],
                                a_max=1 + config["CLIP_EPS"],
                            )
                        ) * gae
                        drift = nn.relu(drift - 1e-4 - nn.relu(ppo_drift))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-generations", type=int, default=800)
    parser.add_argument("--envs", nargs="+", required=False, default=["breakout"])
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--popsize", type=int, default=64)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--save-every-k", type=int, default=8)
    parser.add_argument("--noise-level", type=float, default=0.03)
    parser.add_argument("--wandb-name", type=str, default="OPEN")
    parser.add_argument("--sigma-decay", type=float, default=0.99)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--hsize", type=int, default=128)

    args = parser.parse_args()

    pmap = jax.local_device_count() > 1

    evo_config = {
        "ENV_NAME": args.envs,
        "POPULATION_SIZE": args.popsize,
        "NUM_GENERATIONS": args.num_generations,
        "NUM_ROLLOUTS": args.num_rollouts,
        "SAVE_EVERY_K": args.save_every_k,
        "NOISE_LEVEL": args.noise_level,
        "PMAP": pmap,
        "LR": args.lr,
        "num_GPUs": jax.local_device_count(),
        "project": "lpo_bb",
    }

    all_configs = {k: all_configs[k] for k in evo_config["ENV_NAME"]}

    save_loc = "save_files/lpo_bb"
    os.makedirs(save_loc, exist_ok=True)
    save_dir = f"{save_loc}/{str(datetime.now()).replace(' ', '_')}"
    os.mkdir(f"{save_dir}")

    popsize = args.popsize
    num_generations = args.num_generations
    num_rollouts = args.num_rollouts
    save_every_k_gens = args.save_every_k

    wandb.init(
        project="meta-analysis",

        config=evo_config,
        name=args.wandb_name,
    )
    meta_network = MetaDrift(args.hsize)
    params = meta_network.init(jax.random.PRNGKey(0), jnp.zeros((8,)))
    param_reshaper = ParameterReshaper(params)

    def make_rollout(train_fn):
        def single_rollout(rng_input, meta_params):
            params, metrics = train_fn(meta_params, rng_input)

            fitness = metrics["returned_episode_returns"][-1]
            return (fitness, metrics["returned_episode_returns"][-1])

        vmap_rollout = jax.vmap(single_rollout, in_axes=(0, None))
        rollout = jax.jit(jax.vmap(vmap_rollout, in_axes=(0, param_reshaper.vmap_dict)))

        if evo_config["PMAP"]:
            rollout = jax.pmap(rollout)

        return rollout

    for k in all_configs.keys():
        all_configs[k]["NUM_UPDATES"] = (
            all_configs[k]["TOTAL_TIMESTEPS"]
            // all_configs[k]["NUM_STEPS"]
            // all_configs[k]["NUM_ENVS"]
        )

    rollouts = {k: make_rollout(make_train(v)) for k, v in all_configs.items()}

    rng = jax.random.PRNGKey(42)
    strategy = OpenES(
        popsize=popsize,
        num_dims=param_reshaper.total_params,
        opt_name="adam",
        lrate_init=evo_config["LR"],
        sigma_init=evo_config["NOISE_LEVEL"],
        sigma_decay=args.sigma_decay,
        lrate_decay=args.lr_decay,
    )
    es_params = strategy.default_params

    es_logging = ESLog(
        pholder_params=params, num_generations=num_generations, top_k=5, maximize=True
    )
    log = es_logging.initialize()
    fit_shaper = FitnessShaper(
        centered_rank=True, z_score=False, w_decay=0.0, maximize=True
    )
    fit_shaper_multi = FitnessShaper(
        centered_rank=True, z_score=False, w_decay=0.0, maximize=False
    )

    state = strategy.initialize(rng, es_params)

    most_neg = {env: 0 for env in evo_config["ENV_NAME"]}

    fit_history = []
    for gen in tqdm(range(num_generations)):
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        x, state = jax.jit(strategy.ask)(rng_ask, state, es_params)

        reshaped_params = param_reshaper.reshape(x)
        fit_info = {}

        fit_info[f"meta learning rate"] = state.opt_state.lrate
        fit_info[f"meta noise sigma"] = state.sigma

        all_fitness = []

        for env in args.envs:
            rng, rng_eval = jax.random.split(rng)

            all_configs[env]["VISUALISE"] = False
            rollout = rollouts[env]

            batch_rng = jax.random.split(rng_eval, num_rollouts)
            batch_rng = jnp.tile(batch_rng, (args.popsize, 1, 1))

            if pmap:
                batch_rng_pmap = jnp.reshape(
                    batch_rng, (jax.local_device_count(), -1, num_rollouts, 2)
                )
                fitness, unreg_fitness = rollout(batch_rng_pmap, reshaped_params)

                fitness = fitness[..., -1, :].mean(-1)
                fitness = fitness.reshape(-1, evo_config["NUM_ROLLOUTS"]).mean(axis=1)

            else:
                batch_rng = jnp.reshape(batch_rng, (-1, num_rollouts, 2))
                fitness, unreg_fitness = rollout(batch_rng, reshaped_params)
                fitness = fitness.mean(axis=1)

            print(f"fitness:       {fitness}")

            fit_info[f"{env}/fitness_notnorm_{env}"] = jnp.mean(fitness)
            fit_info[f"{env}/best_fitness_{env}"] = jnp.max(fitness)
            fit_info[f"{env}/worst_fitness{env}"] = jnp.min(fitness)

            fitness = jnp.nan_to_num(fitness, nan=-100000)
            print(f"mean fitness_{env}  =   {jnp.mean(fitness):.3f}")

            fit_re = fit_shaper.apply(x, fitness)

            print(f"fitness_spread at gen {gen} is {fitness.max()-fitness.min()}")
            log = es_logging.update(log, x, fitness)
            print(
                f"Generation: {gen}, Best: {log['log_top_1'][gen]}, Fitness: {fitness.mean()}"
            )
            fit_history.append(fitness.mean())

            fitness_var = jnp.var(fitness)
            dispersion = fitness_var / jnp.abs(fitness.mean())

            param_sum = state.mean.sum()
            param_abs_sum = jnp.abs(state.mean).sum()
            param_abs_mean = jnp.abs(state.mean).mean()
            fitness_spread = fitness.max() - fitness.min()

            fit_norm = fitness / all_configs[env]["PPO_TEMP"]

            mean_norm = fit_norm.mean()

            wandb.log(
                {
                    f"{env}/avg_fitness": fitness.mean(),
                    f"{env}/fitness_histo_{env}": wandb.Histogram(fitness, num_bins=16),
                    f"{env}/fitness_spread_{env}": fitness_spread,
                    f"{env}/fitness_variance": fitness_var,
                    f"{env}_norm_fit": mean_norm,
                    f"{env}_norm_histo": wandb.Histogram(fit_norm, num_bins=16),
                    "dispersion_coeff": dispersion,
                    **fit_info,
                },
                step=gen,
            )

            all_fitness.append(fit_norm)

        fitnesses = jnp.stack(all_fitness, axis=0)

        fitnesses_mean = jnp.mean(fitnesses, axis=0)
        wandb.log(
            {
                "average normalised fit": fitnesses_mean.mean(),
                "average_histo": wandb.Histogram(fitnesses_mean, num_bins=16),
            },
            step=gen,
        )

        fitness_rerank = fit_shaper.apply(x, fitnesses_mean)

        state = jax.jit(strategy.tell)(x, fitness_rerank, state, es_params)

        if gen % save_every_k_gens == 0:
            print("SAVING!")
            jnp.save(osp.join(save_dir, f"curr_param_{gen}.npy"), state.mean)
            np.save(osp.join(save_dir, f"fit_history.npy"), np.array(fit_history))

            wandb.save(
                osp.join(save_dir, f"curr_param_{gen}.npy"),
                base_path=save_dir,
            )

            plot_envs = args.envs

            time.sleep(1)
            eval_func(
                meta_params=state.mean,
                hsize=args.hsize,
                envs=plot_envs,
                num_runs=8,
                iteration=gen,
            )

            plt.close("all")
