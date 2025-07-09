import argparse
import json
import os
import time
from datetime import datetime

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import openai
from openai import OpenAI

import wandb
from algorithms.evaluation.eval_lpo_llm import eval_func

API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


def init_archive(config, args):
    archive = []

    archive.append(
        {
            "code": """
def ppo_clip(
    ratio: float,
    advantage: float,
    epsilon: float
) -> jnp.ndarray:
    ratio_clip = ratio - jnp.clip(ratio, a_min = 1-epsilon, a_max = 1+epsilon)
    ratio_adv = ratio_clip * advantage
    drift = nn.relu(ratio_adv)
    return drift
        """,
        }
    )

    namespace = {}
    exec(archive[0]["code"], globals(), namespace)
    names = list(namespace.keys())
    func = namespace[names[0]]
    print(config)

    fitness, _ = eval_func(config["envs"], func, iteration=0)
    archive[0]["fitness"] = fitness
    return archive


def validate_code(code: str, args) -> bool:
    # Run code through test
    try:
        # Namespace dictionary to hold the execution context
        namespace = {}

        # Execute the function definition string within the provided namespace
        exec(code, globals(), namespace)

        names = list(namespace.keys())
        if len(names) != 1:
            return False, f"{len(names)} things in namespace. Please only provide 1"
        func = namespace[names[0]]
        if not callable(func):
            return False, f"{func} is not callable"

        # Need to test for Nans etc. and make sure the shape is correct
        rng = jax.random.PRNGKey(0)
        rng = jax.random.split(rng, 2)

        r = jax.random.normal(rng[0], (10,)) * 0.1 + 1
        A = jax.random.normal(rng[1], (10,))
        epsilon = 0.2

        update = func(r, A, epsilon)

        # Check for NaNs in the output
        if jnp.isnan(update).any():
            return False, "Update contains NaNs"

        # Check the shape of the output
        if update.shape != (10,):
            return (
                False,
                f"Expected update shape to be per input (e.g. (10,)), got {update.shape}",
            )

        return True, ""

    except Exception as e:
        print(code)
        print(str(e))
        return False, str(e)


parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=True)
parser.add_argument("--num-generations", type=int, default=30)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--envs", type=str, default=["breakout"], nargs="+")

if __name__ == "__main__":
    args = parser.parse_args()

    client = OpenAI()

    now = str(datetime.now()).replace(" ", "_")
    save_dir = f"save_files/llm_runs/{now}"
    os.makedirs("save_files/llm_runs", exist_ok=True)

    os.mkdir(save_dir)

    config = {
        "NUM_GENERATIONS": args.num_generations,
        "SAVE_DIR": save_dir,
        "NOW": now,
        "RESUME": args.resume,
        "envs": args.envs,
    }

    if args.wandb:
        run = wandb.init(
            name="lpo_llm",
            project="meta-analysis",
            config=config,
        )

    system_prompt = f"""
You are a machine learning researcher who is designing a new drift function for reinforcement learning. When you respond, output a JSON where the first key ("thought") corresponds to your thought process when designing the next function. The second key ("name") corresponds to the name of your next function. Finally, the last key ("code") corresponds to the exact python code that you would like to try. Here is an example:

"thought": "Based on the previous outputs, I should try to tanh the function.",
"name": "tanh_clip",
"code": "def tanh_clip(
    ratio: float,
    advantage: float,
    epsilon: float
) -> jnp.ndarray:
    ratio_clip = jnp.tanh(ratio - jnp.clip(ratio, a_min = 1-epsilon, a_max = 1+epsilon))
    ratio_adv = ratio_clip * advantage
    drift = nn.relu(ratio_adv)
    return drift"

You are deeply familiar with drift functions for reinforcement learning from the literature. Be creative and reference prior literature when possible.

You must use the exact function interface used above. Your function should return only the function value, which will be applied to limit large changes to the policy. Feel free to define extra hyperparameters within your function as constants. Do not make them attributes of self. You may use whichever jax functions you want, including logic functions if appropriate.

Drift functions use the ratio and advantage to limit changes to the policy after updating. To be a valid drift function, the function must be non-negative everywhere, zero at identity (when r=1) and have a gradient of zero with respect to r at r=1. It can be easier to guarantee this by using functions of (r-1) or jnp.log(r).
        `r' is the ratio of the new policy to a reference policy, which is the previous policy in this case.
        `A' is the GAE advantage estimate of the policy.
        `epsilon' is the clip epsilon value used in PPO.
        You may also use branching functions such as jax.lax.cond or take the maximum of two values.

The user will then return to you a fitness that corresponds to the performance of the resulting model on a downstream task. Your goal is to maximize performance.
"""
    archive = init_archive(config, args)
    first_prompt = f"""
Here are some results we've obtained: \n{archive}

Please generate the next one.
"""
    if not args.resume:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt},
        ]
    else:
        messages = json.load(open(args.resume, "r"))
        assert messages[-1]["role"] == "user", "Last message must be user"

    t0 = time.time()
    t_start, t_completion, t_train_start, t_train_end, t_eval_end = t0, t0, t0, t0, t0
    for i in range(args.num_generations):
        if i > 0 and args.wandb:
            columns = ["generation", "thought", "function", "fitness", "next prompt"]
            log_table = wandb.Table(columns=columns)
            if "thought" in out and "code" in out:
                log_table.add_data(i, out["thought"], out["code"], fitness, next_prompt)
            else:
                log_table.add_data(i, "", "", fitness, next_prompt)

            wandb.log(
                {
                    "fitness": fitness,
                    "generation": i,
                    "generation_time": time.time() - t_start,
                    "code_time": t_completion - t_start,
                    "train_time": t_train_end - t_train_start,
                    f"table_{i}": log_table,
                },
                step=i,
            )
        t_start = time.time()
        # GENERATE CODE
        for _ in range(API_MAX_RETRY):
            try:
                completion = client.chat.completions.create(
                    model="o3-mini",
                    messages=messages,
                    n=1,
                    response_format={"type": "json_object"},
                ).choices[0]
                break
            except openai.OpenAIError as e:
                print(type(e), e)
                time.sleep(API_RETRY_SLEEP)

        t_completion = time.time()
        messages.append(completion.message.to_dict())
        with open(f"{save_dir}/messages.json", "w") as f:
            json.dump(messages, f, indent=4)
        out = json.loads(completion.message.content)

        # VALIDATE CODE
        valid, error = validate_code(out["code"], args)
        if not valid:
            next_prompt = (
                f"Code not valid. Error:\n{error}\nPlease generate the next one."
            )
            messages.append({"role": "user", "content": next_prompt})
            fitness = -10000
            print("CODE NOT VALID")
            continue
        t_train_start = time.time()

        if jax.local_device_count() > 1:
            config["pmap"] = True
        else:
            config["pmap"] = False

        namespace = {}
        exec(out["code"], globals(), namespace)
        names = list(namespace.keys())
        func = namespace[names[0]]

        val, error = eval_func(config["envs"], func, iteration=i)
        if not val:
            next_prompt = (
                f"Training failed. Error:\n{error}\nPlease generate the next one."
            )
            messages.append({"role": "user", "content": next_prompt})
            fitness = -10000
            print("FAILED TRAINING")
            continue
        t_train_end = time.time()

        next_prompt = f"Fitness: {val}.\nPlease generate the next one."
        messages.append({"role": "user", "content": next_prompt})
        fitness = val

        archive.append({"code": out["code"], "fitness": val})
        arch_table = wandb.Table(
            columns=["code", "fitness"], data=[list(d.values()) for d in archive]
        )
        wandb.log({"archive": arch_table}, step=i)

        plt.close("all")

    columns = ["generation", "thought", "function", "fitness", "next prompt"]
    log_table = wandb.Table(columns=columns)
    if "thought" in out and "code" in out:
        log_table.add_data(i, out["thought"], out["code"], fitness, next_prompt)
    else:
        log_table.add_data(i, "", "", fitness, next_prompt)
    print("logging")
    wandb.log(
        {
            "fitness": fitness,
            "generation": i,
            "generation_time": time.time() - t_start,
            "code_time": t_completion - t_start,
            "train_time": t_train_end - t_train_start,
            f"table_{i}": log_table,
        },
        step=i,
    )
    arch_columns = ["code", "fitness"]
    arch_table = wandb.Table(
        columns=["code", "fitness"], data=[list(d.values()) for d in archive]
    )
    wandb.log({"archive": arch_table}, step=i)

    print(messages)
