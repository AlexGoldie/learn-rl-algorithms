import argparse
import os
import warnings

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from evosax import ParameterReshaper
from pysr import PySRRegressor, jl

import wandb
from algorithms.evaluation.eval_lpo_symbolic import eval_func
from algorithms.learning_algorithms.symbolic_distillation.utils import (
    create_jax_function,
)

api = wandb.Api()


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def transform_data_DPO(ratio, advantage):
    log_ratio = jnp.log(ratio)
    r1 = ratio - 1
    meta_inputs = jnp.stack([r1, r1 * r1, log_ratio, log_ratio * log_ratio], axis=-1)
    meta_inputs = jnp.concatenate(
        [meta_inputs, meta_inputs * advantage[..., None]], axis=-1
    )
    return meta_inputs


class Distilled_DPO(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128, use_bias=False)(x)
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

        # drift_fn_grad = jax.grad(drift_fn_sum)
        policy_loss_grad = jax.grad(policy_loss)

        return policy_loss_grad(ratio, advantage)


def get_data_lpo():
    advantages = jnp.arange(-512, 512) / (128)
    ratios = jnp.exp(jnp.arange(-512, 512) / (512))
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

    target = drift_fn(ratio, advantage)

    return (jnp.log(ratio), ratio - 1, advantage), target


def vis_drift(iter, func, meta_params, meta_network, title=""):
    fig = plt.figure()

    eps = 0.2

    advantages = jnp.arange(-512, 512) / (64)
    ratios = jnp.exp(jnp.arange(-512, 512) / (512))
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

    def make_drift_fn(drift_apply):
        def drift_fn(ratio, advantage):
            log_ratio = jnp.log(ratio)
            r1 = ratio - 1

            inps = jnp.stack([log_ratio, r1, advantage], axis=-1)

            out = drift_apply(inps)

            drift = nn.relu(out - 1e-6)
            return drift

        def drift_fn_sum(ratio, advantage):
            return drift_fn(ratio, advantage).sum()

        def policy_loss(ratio, advantage):
            drift = drift_fn(ratio, advantage)
            return -(ratio * advantage - drift).mean()

        drift_fn_grad = jax.grad(drift_fn_sum)
        policy_loss_grad = jax.grad(policy_loss)

        return drift_fn, drift_fn_grad, policy_loss_grad

    advantages = jnp.arange(-512, 512) / (128)
    ratios = jnp.exp(jnp.arange(-512, 512) / (512))
    ratio_and_advantage = jnp.transpose(
        jnp.array(
            [jnp.tile(ratios, len(advantages)), jnp.repeat(advantages, len(ratios))]
        )
    )
    ratio = ratio_and_advantage[:, 0]
    advantage = ratio_and_advantage[:, 1]

    drift_fn, drift_fn_grad, policy_loss_grad = make_drift_fn(func)

    drift_map = drift_fn(ratio, advantage)

    drift_map = drift_map.reshape(1024, 1024)

    ppo_drift_map = ppo_drift(ratio, advantage)
    ppo_drift_map = ppo_drift_map.reshape(1024, 1024)

    loss_grad_map = policy_loss_grad(ratio, advantage)
    loss_grad_map = loss_grad_map.reshape(1024, 1024)

    @plt.FuncFormatter
    def fake_log(x, pos):
        "The two args are the value and tick position"
        return r"$e^{%0.2f}$" % (x)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 10)
    )  # Create two subplots stacked vertically

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
            drift = nn.relu(out - 1e-6)
            return drift

        def drift_fn_sum(ratio, advantage):
            return drift_fn(ratio, advantage).sum()

        def policy_loss(ratio, advantage):
            drift = drift_fn(ratio, advantage)
            return -(ratio * advantage - drift).mean()

        drift_fn_grad = jax.grad(drift_fn_sum)
        policy_loss_grad = jax.grad(policy_loss)

        return drift_fn, drift_fn_grad, policy_loss_grad

    drift_fn2, drift_fn_grad2, policy_loss_grad2 = make_drift_fn_network(
        meta_network.apply, meta_params
    )

    loss_grad_map2 = policy_loss_grad2(ratio, advantage)
    loss_grad_map2 = loss_grad_map2.reshape(1024, 1024)

    # Plot for loss_grad_map on ax1
    img1 = ax1.imshow(
        -jnp.flip(loss_grad_map, 0),
        extent=[
            jnp.log(ratios).min().item(),
            jnp.log(ratios).max().item(),
            advantages.min().item(),
            advantages.max().item(),
        ],
        aspect="auto",
        cmap=plt.cm.coolwarm,
        vmin=-1e-5,
        vmax=1e-5,
    )
    fig.colorbar(img1, ax=ax1)  # Add a colorbar to the first subplot
    ax1.xaxis.set_major_formatter(fake_log)
    ax1.set_title("Curent Best")
    ax1.set_xlabel("Log Ratio")
    ax1.set_ylabel("Advantage")
    img2 = ax2.imshow(
        -jnp.flip(loss_grad_map2, 0),
        extent=[
            jnp.log(ratios).min().item(),
            jnp.log(ratios).max().item(),
            advantages.min().item(),
            advantages.max().item(),
        ],
        aspect="auto",
        cmap=plt.cm.coolwarm,
        vmin=-1e-5,
        vmax=1e-5,
    )
    fig.colorbar(img2, ax=ax2)  # Add a colorbar to the second subplot
    ax2.xaxis.set_major_formatter(fake_log)
    ax2.set_title("Derivative of LPO Objective (Map 2)")
    ax2.set_xlabel("Log Ratio")
    ax2.set_ylabel("Advantage")
    os.makedirs("save_files/dpo", exist_ok=True)
    fig.savefig(
        f"save_files/dpo/plots.png",
        format="png",
        bbox_inches="tight",
    )
    import time

    time.sleep(1)
    wandb.log({f"heatmap/dpo_map": wandb.Image(f"save_files/dpo/plots.png")}, step=iter)


# =-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", required=False, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--exp-num", type=int, default=None)
    parser.add_argument("--file-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--base-hsize", type=int, default=128)
    parser.add_argument("--grad-loss", default=False, action="store_true")

    args = parser.parse_args()

    if args.envs == None:
        envs = ["breakout", "cartpole", "asterix", "ant", "spaceinvaders", "freeway"]
    else:
        envs = args.envs

    config = {
        "envs": envs,
        "target": "lpo",
        "exp_name": args.exp_name,
        "exp_num": args.exp_num,
    }

    wandb.init(
        project="meta-analysis",
        name="lpo_symbolic",
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

    meta_network = Distilled_DPO()
    test_case = jnp.zeros((2, 8))
    pholder = meta_network.init(jax.random.PRNGKey(0), test_case)
    param_reshaper = ParameterReshaper(pholder)

    meta_params = param_reshaper.reshape_single(params)

    rng = jax.random.PRNGKey(100)

    if jax.local_device_count() > 1:
        pmap = True
    else:
        pmap = False

    jl.seval(
        """
using Pkg
Pkg.add("Zygote")
"""
    )
    jl.seval("using Zygote")

    # we provide code to compare with the gradient rather than function outputs if desired
    jl.seval(
        """function custom_loss_grad(tree, dataset::Dataset{T,L}, options, idx) where {T,L}

idx = isnothing(idx) ? (1:size(dataset.X, 2)) : idx
X = copy(dataset.X[:, idx])  # shape (n, x)
y = copy(dataset.y[idx])  # shape (n)
# First evaluate the tree
features = X
r = features[2,:]
# println("features:   ", features)

test = [0. 0. 0.; 0. 0. 0.; 1. -1. 0.5]
evaluation, complete = eval_tree_array(tree, test, options)
if any(abs.(evaluation) .!= 0.)
    return L(Inf)
end

r = r .+ 1

# Now compute gradients of the ReLU output with respect to features

# use the chain rule!

# u = (r-1), v = log(r)
dudr = 1 #u = r-1 -> dudr = 1
evaluation, gradients_u, complete = eval_diff_tree_array(tree, features, options, 2)
dvdr = 1 ./ r #v = log(r) -> dvdr = 1/r

# println("dvdr:   ", dvdr)
evaluation, gradients_v, complete = eval_diff_tree_array(tree, features, options, 1)

# println("tree:    ",    tree)
gradients = gradients_u .+ gradients_v .* dvdr
evaluation, complete = eval_tree_array(tree, features, options)

relu_mask = (evaluation .- 1e-6) .> 0  # Element-wise mask: 1 where evaluation > 0, else 0

gradients = gradients .* relu_mask

# Compute L2 loss between the gradient and target values
loss = sum((gradients .- y).^2)/length(idx)

# loss = loss.^0.5

return loss
end"""
    )

    # We also have a loss which checks that we satisfy the LPO constraints
    jl.seval(
        """function custom_loss_2(tree, dataset::Dataset{T,L}, options, idx) where {T,L}

idx = isnothing(idx) ? (1:size(dataset.X, 2)) : idx
X = copy(dataset.X[:, idx])  # shape (n, x)
y = copy(dataset.y[idx])  # shape (n)
# First evaluate the tree
features = X
r = features[2,:]
# println("features:   ", features)

test = [0. 0. 0.; 0. 0. 0.; 1. -1. 0.5]
evaluation, complete = eval_tree_array(tree, test, options)
if any(abs.(evaluation) .!= 0.)
    return L(Inf)
end

out, complete = eval_tree_array(tree, features, options)

# Compute L2 loss between the gradient and target values
loss = sum((out .- y).^2)/length(idx)

# loss = loss.^0.5

return loss
end"""
    )

    if args.grad_loss:
        loss_func = "custom_loss_grad"
    else:
        loss_func = "custom_loss_2"

    model = PySRRegressor(
        maxsize=40,
        weight_optimize=0.001,
        turbo=True,
        niterations=1,  # < Increase me for better results
        binary_operators=["*", "/", "max", "min", "+", "-"],
        unary_operators=[
            "square",
            "neg",
            "tanh",
            "abs",
        ],
        # ^ Define operator for SymPy as well
        loss_function=loss_func,  # to compare gradient, use custom_loss_grad
        warm_start=True,
        print_precision=3,
        maxdepth=8,
        precision=64,
        crossover_probability=0.01,
        batch_size=5000,
        batching=True,
    )

    epochs = args.epochs

    rng = jax.random.PRNGKey(0)
    rng, rng_ = jax.random.split(rng)
    train_inp, train_targ = get_data_lpo()
    test_inp, test_targ = get_data_lpo()
    inp = jnp.stack(train_inp, -1)
    test_inp = jnp.stack(test_inp, -1)
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
        vis_drift(
            iter=i,
            func=jax_func,
            meta_params=meta_params,
            meta_network=meta_network,
        )

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
