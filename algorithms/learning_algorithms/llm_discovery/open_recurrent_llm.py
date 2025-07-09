import argparse
import json
import os
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import openai
from openai import OpenAI

import wandb
from algorithms.evaluation.eval_open_recurrent_llm import eval_func

API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


def init_archive(config):
    archive = []

    # Need to make fitnesses and initialise archive

    archive.append(
        {
            "code": """
def Adam(
    p: jnp.ndarray,
    m_0_1: jnp.ndarray,
    m_0_5: jnp.ndarray,
    m_0_9: jnp.ndarray,
    m_0_99: jnp.ndarray,
    m_0_999: jnp.ndarray,
    m_0_9999: jnp.ndarray,
    l_p: jnp.ndarray,
    b_p: jnp.ndarray,
    t_p: jnp.ndarray,
    dorm: jnp.ndarray,
    g: jnp.ndarray,
    rand: jnp.ndarray,
    lr: float,
    iteration: float,
    var: jnp.ndarray
) -> jnp.ndarray:

    var = (1-0.999) * jnp.square(g) + 0.999 * var
    var_hat = var / (1-0.999**iteration)

    m_hat = m_0_9 / (1-0.9**iteration)

    adam = m_hat / jnp.sqrt(var_hat + 1e-8)

    update = adam * lr
    return update, var
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

    print(archive)
    return archive


def validate_code(code: str) -> bool:
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
        rng = jax.random.split(rng, 8)

        p = jax.random.normal(rng[0], (10,))
        g = jax.random.normal(rng[1], (10,))
        m_0_1 = jax.random.normal(rng[2], (10,))
        m_0_5 = jax.random.normal(rng[3], (10,))
        m_0_9 = jax.random.normal(rng[4], (10,))
        m_0_99 = jax.random.normal(rng[5], (10,))
        m_0_999 = jax.random.normal(rng[6], (10,))
        m_0_9999 = jax.random.normal(rng[7], (10,))
        lr = 1
        l_p = jax.random.normal(rng[7], (10,))
        b_p = jax.random.normal(rng[7], (10,))
        t_p = jax.random.normal(rng[7], (10,))
        dorm = jax.random.normal(rng[7], (10,))
        rand = jax.random.normal(rng[7], (10,))
        iteration = jnp.ones(
            10,
        )
        var = jnp.ones_like(p)

        update, carry = func(
            p,
            m_0_1,
            m_0_5,
            m_0_9,
            m_0_99,
            m_0_999,
            m_0_9999,
            l_p,
            b_p,
            t_p,
            dorm,
            g,
            rand,
            lr,
            iteration,
            var,
        )

        # Check for NaNs in the output
        if jnp.isnan(update).any():
            return False, "Update contains NaNs"

        if jnp.isnan(carry).any():
            return False, "var contains NaNs"

        # Check the shape of the output
        if update.shape != (10,):
            return (
                False,
                f"Expected update shape to be per input (e.g. (10,)), got {update.shape}",
            )

        # Check the shape of the output
        if carry.shape != (10,):
            return (
                False,
                f"Expected var shape to be per input (e.g. (10,)), got {update.shape}",
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
            name=f"open_rec_llm",
            project="meta-analysis",
            config=config,
        )

    system_prompt = """
You are a machine learning researcher who is designing a new optimisation algorithm for reinforcement learning. When you respond, output a JSON where the first key ("thought") corresponds to your thought process when designing the next function. The second key ("name") corresponds to the name of your next function. Finally, the last key ("code") corresponds to the exact python code that you would like to try. Here is an example:

{"thought": "Based on the previous outputs, I should try using making the bias correction relative to each batch.",
"name": "Adam",
"code": "def Adam(
    p: jnp.ndarray,
    m_0_1: jnp.ndarray,
    m_0_5: jnp.ndarray,
    m_0_9: jnp.ndarray,
    m_0_99: jnp.ndarray,
    m_0_999: jnp.ndarray,
    m_0_9999: jnp.ndarray,
    l_p: jnp.ndarray,
    b_p: jnp.ndarray,
    t_p: jnp.ndarray,
    dorm: jnp.ndarray,
    g: jnp.ndarray,
    rand: jnp.ndarray,
    lr: float,
    iteration: float,
    var: jnp.ndarray
) -> jnp.ndarray:

    var = (1-0.999) * jnp.square(g) + 0.999 * var
    var_hat = var / (1-0.999**iteration)

    m_hat = m_0_9 / (1-0.9**iteration)

    adam = m_hat / jnp.sqrt(var_hat + 1e-8)

    update = adam * lr

    return update, var"
}

You are deeply familiar with optimisation for reinforcement learning from the literature. Be creative and reference prior literature when possible.

You must use the exact function interface used above. Your function should return the update value, which will be applied separately to the parameters, and the var value, which will be used as a momentum variable between iterations. Feel free to define extra hyperparameters within your function as constants. Do not make them attributes of self. You may use whichever jax functions you want, including logic functions if appropriate. Note that `lr' is tuned per environment, and is annealed over the course of training.

Optimisation algorithms use the gradient, and other inputs, to calculate updates to the parameters of a neural network. Here, we provide a number of additional inputs which have previously been found to be helpful in optimisation for reinforcement learning. You may choose to use as many or as few inputs as you would like.
`p` refers to the current value of the parameter being optimised.
`g` refers to the gradient of the loss function with respect to the parameter.
`m_x_y` refers to the historic momentum of the gradient. This is calculated as m_x_y = (x.y) * g + (1-x.y) * m_x_y.
`dorm` refers to the dormancy of the neuron which the parameter is going into.
`l_p` is the layer proportion, and refers to how deep a parameter is through a neural network. It starts at 0. in the first layer, and increases to 1. in the final layer.
`b_p` is the batch proportion, and refers to how far through the total number of epochs with a fixed batch of data training is.
`t_p` is the training proportion, and refers to how far training is through the full horizon.
`dorm` is the dormancy, and refers to the how much of a layer's activation comes from a specific neuron. It is measured between 0. and the number of neurons in a layer.
`rand` is a random, normally distributed value, which can be applied for stochasticity.
`iteration` is the total iteration count.
`var` is a recurrent variable which is passed between training iterations. You may use it to store any information which might be useful for historical conditioning.

The user will then return to you a fitness that corresponds to the performance of the resulting model on a downstream task. Your goal is to maximize performance.
"""
    archive = init_archive(config)
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
            print(fitness)
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
        t_start = time.time()
        # GENERATE CODE
        for _ in range(API_MAX_RETRY):
            try:
                completion = client.chat.completions.create(
                    model="o3-mini",
                    messages=messages,
                    # max_tokens=2048,
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
        valid, error = validate_code(out["code"])
        if not valid:
            next_prompt = (
                f"Code not valid. Error:\n{error}\nPlease generate the next one."
            )
            messages.append({"role": "user", "content": next_prompt})
            fitness = -10000
            print("CODE NOT VALID")
            continue
        t_train_start = time.time()

        # TRAIN GPO
        if jax.local_device_count() > 1:
            config["pmap"] = True
        else:
            config["pmap"] = False

        namespace = {}
        exec(out["code"], globals(), namespace)
        names = list(namespace.keys())
        func = namespace[names[0]]

        val, error = eval_func(config["envs"], func, iteration=i)
        print(val)
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
