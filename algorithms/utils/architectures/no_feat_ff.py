from typing import Any, Optional

import flax
import flax.linen as nn
import gin
import jax
import jax.numpy as jnp
from jax import lax

from algorithms.utils import base as opt_base
from algorithms.utils.learned_optimization.learned_optimization import tree_utils
from algorithms.utils.learned_optimization.learned_optimization.learned_optimizers import (
    common,
)

PRNGKey = jnp.ndarray


def _second_moment_normalizer(x, axis, eps=1e-5):
    return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def iter_proportion(iterations, total_its=100000):
    f32 = jnp.float32

    return iterations / f32(total_its)


@flax.struct.dataclass
class mlpState:
    params: Any
    rolling_features: common.MomAccumulator
    iteration: jnp.ndarray
    state: Any
    carry: Any


hidden_size = 32


@gin.configurable
class mlp_no_feat(opt_base.LearnedOptimizer):
    def __init__(
        self,
        exp_mult=0.001,
        step_mult=0.001,
        hidden_size=16,
    ):
        super().__init__()
        self._step_mult = step_mult
        self._exp_mult = exp_mult

        self._mod = nn.Sequential(
            [
                nn.Dense(hidden_size),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(hidden_size),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(2),
            ]
        )

    def init(self, key: PRNGKey) -> opt_base.MetaParams:
        # There are 19 features used as input. For now, hard code this.
        key = jax.random.split(key, 5)

        return {
            "params": self._mod.init(key[0], jnp.zeros([8])),
        }

    def opt_fn(
        self, theta: opt_base.MetaParams, is_training: bool = False
    ) -> opt_base.Optimizer:
        decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])

        mod = self._mod
        exp_mult = self._exp_mult
        step_mult = self._step_mult

        theta_mlp = theta["params"]

        class _Opt(opt_base.Optimizer):
            """Optimizer instance which has captured the meta-params (theta)."""

            def init(
                self,
                params: opt_base.Params,
                model_state: Any = None,
                num_steps: Optional[int] = None,
                key: Optional[PRNGKey] = None,
            ) -> mlpState:
                """Initialize inner opt state."""

                param_tree = jax.tree_util.tree_structure(params)

                keys = jax.random.split(key, param_tree.num_leaves)
                keys = jax.tree_util.tree_unflatten(param_tree, keys)

                return mlpState(
                    params=params,
                    state=model_state,
                    rolling_features=common.vec_rolling_mom(decays).init(params),
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                    carry=None,
                )

            def update(
                self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                opt_state_actor: mlpState,
                crit_opt_state: mlpState,
                grad: Any,
                activations: float,
                key: Optional[PRNGKey] = None,
                training_prop=0,
                batch_prop=0,
                config=None,
                layer_props=None,
                model_state: Any = None,
                mask=None,
            ) -> mlpState:
                next_rolling_features_actor = common.vec_rolling_mom(decays).update(
                    opt_state_actor.rolling_features, grad["actor"]
                )

                next_rolling_features_critic = common.vec_rolling_mom(decays).update(
                    crit_opt_state.rolling_features, grad["critic"]
                )

                rolling_features = {
                    "actor": next_rolling_features_actor.m,
                    "critic": next_rolling_features_critic.m,
                }

                def _update_tensor(p, g, mom):

                    # this doesn't work with scalar parameters, so let's reshape.
                    if not p.shape:
                        p = jnp.expand_dims(p, 0)

                        # use gradient conditioning (L2O4RL)
                        g = jnp.expand_dims(g, 0)

                        mom = jnp.expand_dims(mom, 0)
                        did_reshape = True
                    else:
                        did_reshape = False

                    inps = []

                    batch_g = jnp.expand_dims(g, axis=-1)

                    # **NO CURRENT GRADIENT**
                    # feature consisting of raw parameter values
                    batch_p = jnp.expand_dims(p, axis=-1)
                    inps.append(batch_p)

                    # feature consisting of all momentum values
                    inps.append(mom)

                    inp_stack = jnp.concatenate(inps, axis=-1)

                    axis = list(range(len(p.shape)))

                    inp_stack_g = jnp.concatenate([inp_stack, batch_g], axis=-1)

                    inp_stack_g = _second_moment_normalizer(inp_stack_g, axis=axis)

                    output = mod.apply(theta_mlp, inp_stack_g)

                    update = (
                        output[..., 0] * step_mult * jnp.exp(output[..., 1] * exp_mult)
                    )

                    update = update.reshape(p.shape)

                    return (
                        update,
                        update,
                    )

                full_params = {
                    "actor": opt_state_actor.params,
                    "critic": crit_opt_state.params,
                }
                param_tree = jax.tree_util.tree_structure(full_params)

                keys = jax.random.split(key, param_tree.num_leaves)
                keys = jax.tree_util.tree_unflatten(param_tree, keys)

                activations = jax.tree_util.tree_flatten(activations)[0]

                activations = jax.tree_util.tree_unflatten(param_tree, activations)

                def calc_dormancy(tensor_activations):
                    tensor_activations = tensor_activations + 1e-11
                    total_activations = jnp.abs(tensor_activations).sum(axis=-1)
                    total_activations = jnp.tile(
                        jnp.expand_dims(total_activations, -1),
                        tensor_activations.shape[-1],
                    )
                    dormancy = (
                        tensor_activations
                        / total_activations
                        * tensor_activations.shape[-1]
                    )
                    return dormancy

                updates_carry = jax.tree_util.tree_map(
                    _update_tensor,
                    full_params,
                    grad,
                    rolling_features,
                )

                updates_carry_leaves = jax.tree_util.tree_leaves(updates_carry)
                updates = [
                    updates_carry_leaves[i]
                    for i in range(0, len(updates_carry_leaves), 2)
                ]

                targets = None
                updates = jax.tree_util.tree_unflatten(param_tree, updates)

                # Make update globally 0
                updates_flat = jax.flatten_util.ravel_pytree(updates)[0]
                update_mean = updates_flat.mean()
                update_mean = jax.tree_util.tree_unflatten(
                    param_tree, jnp.tile(update_mean, param_tree.num_leaves)
                )

                updates = jax.tree_util.tree_map(
                    lambda x, mu: x - mu, updates, update_mean
                )

                def param_update(p, update):
                    new_param = p - update

                    return new_param

                next_params = jax.tree_util.tree_map(param_update, full_params, updates)

                next_opt_state_actor = mlpState(
                    params=tree_utils.match_type(
                        next_params["actor"], opt_state_actor.params
                    ),
                    rolling_features=tree_utils.match_type(
                        next_rolling_features_actor, opt_state_actor.rolling_features
                    ),
                    iteration=opt_state_actor.iteration + 1,
                    state=model_state,
                    carry=None,
                )

                next_opt_state_critic = mlpState(
                    params=tree_utils.match_type(
                        next_params["critic"], crit_opt_state.params
                    ),
                    rolling_features=tree_utils.match_type(
                        next_rolling_features_critic, crit_opt_state.rolling_features
                    ),
                    iteration=opt_state_actor.iteration + 1,
                    state=model_state,
                    carry=None,
                )

                return (
                    next_opt_state_actor,
                    next_opt_state_critic,
                    targets,
                )

        return _Opt()
