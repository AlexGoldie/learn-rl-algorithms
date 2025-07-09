"""A learned optimizer taking the form of an MLP with a small amount of noise at its output to incorporate
Parameter Space Noise. This has been designed to work with the same form as the learned optimisers
released by Google.
"""

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
class mlp_open_ff(opt_base.LearnedOptimizer):
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
                nn.Dense(3),
            ]
        )

    def init(self, key: PRNGKey) -> opt_base.MetaParams:
        # There are 19 features used as input. For now, hard code this.
        key = jax.random.split(key, 5)

        return {
            "params": self._mod.init(key[0], jnp.zeros([19])),
        }

    def opt_fn(
        self, theta: opt_base.MetaParams, is_training: bool = False
    ) -> opt_base.Optimizer:
        decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])

        mod = self._mod
        # gru = self._gru
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
                self,
                opt_state_actor: mlpState,
                crit_opt_state: mlpState,
                grad: Any,
                activations: float,
                key: Optional[PRNGKey] = None,
                training_prop=0,
                batch_prop=0,
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

                training_step_feature = training_prop
                batch_feature = batch_prop
                eps1 = 1e-13
                eps2 = 1e-8
                t = opt_state_actor.iteration + 1

                def _update_tensor(p, g, mom, k, dorm, layer_prop, mask):

                    # this doesn't work with scalar parameters, so let's reshape.
                    if not p.shape:
                        p = jnp.expand_dims(p, 0)

                        # use gradient conditioning (L2O4RL)
                        gsign = jnp.expand_dims(jnp.sign(g), 0)
                        glog = jnp.expand_dims(jnp.log(jnp.abs(g) + eps1), 0)

                        mom = jnp.expand_dims(mom, 0)
                        did_reshape = True
                    else:
                        gsign = jnp.sign(g)
                        glog = jnp.log(jnp.abs(g) + eps1)
                        did_reshape = False

                    inps = []
                    inp_g = []

                    batch_gsign = jnp.expand_dims(gsign, axis=-1)
                    batch_glog = jnp.expand_dims(glog, axis=-1)

                    # **NO CURRENT GRADIENT**
                    # feature consisting of raw parameter values
                    batch_p = jnp.expand_dims(p, axis=-1)
                    inps.append(batch_p)

                    # feature consisting of all momentum values
                    momsign = jnp.sign(mom)
                    momlog = jnp.log(jnp.abs(mom) + eps1)
                    inps.append(momsign)
                    inps.append(momlog)

                    inp_stack = jnp.concatenate(inps, axis=-1)

                    axis = list(range(len(p.shape)))

                    inp_stack_g = jnp.concatenate(
                        [inp_stack, batch_gsign, batch_glog], axis=-1
                    )

                    inp_stack_g = _second_moment_normalizer(inp_stack_g, axis=axis)

                    # once normalized, add features that are constant across tensor.
                    # namly the training step embedding.
                    def stack_tensors(feature, input):
                        stacked = jnp.reshape(
                            feature, [1] * len(axis) + list(feature.shape[-1:])
                        )
                        stacked = jnp.tile(stacked, list(p.shape) + [1])
                        return jnp.concatenate([input, stacked], axis=-1)

                    inp = jnp.tile(
                        jnp.reshape(
                            training_step_feature,
                            [1] * len(axis) + list(training_step_feature.shape[-1:]),
                        ),
                        list(p.shape) + [1],
                    )

                    stacked_batch_prop = jnp.tile(
                        jnp.reshape(
                            batch_feature,
                            [1] * len(axis) + list(batch_feature.shape[-1:]),
                        ),
                        list(p.shape) + [1],
                    )

                    layer_prop = jnp.expand_dims(layer_prop, 0)

                    stacked_layer_prop = jnp.tile(
                        jnp.reshape(
                            layer_prop, [1] * len(axis) + list(layer_prop.shape[-1:])
                        ),
                        list(p.shape) + [1],
                    )

                    inp = jnp.concatenate([inp, stacked_layer_prop], axis=-1)

                    inp = jnp.concatenate([inp, stacked_batch_prop], axis=-1)

                    batch_dorm = jnp.expand_dims(dorm, axis=-1)

                    if p.shape != dorm.shape:
                        batch_dorm = jnp.tile(
                            batch_dorm, [p.shape[0]] + len(axis) * [1]
                        )

                    inp = jnp.concatenate([inp, batch_dorm], axis=-1)

                    inp_g = jnp.concatenate([inp_stack_g, inp], axis=-1)

                    # apply the per parameter MLP.
                    output = mod.apply(theta_mlp, inp_g)

                    update_ = (
                        output[..., 0] * step_mult * jnp.exp(output[..., 1] * exp_mult)
                    )
                    update = update_
                    update = (
                        update_
                        + output[..., 2]
                        * mask
                        * jax.random.normal(k, shape=update_.shape)
                        * step_mult
                    )

                    update = update.reshape(p.shape)

                    return (
                        update,
                        update_,
                        mask * jax.random.normal(k, shape=update_.shape),
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

                dormancies = jax.tree_util.tree_map(calc_dormancy, activations)

                updates_carry = jax.tree_util.tree_map(
                    _update_tensor,
                    full_params,
                    grad,
                    rolling_features,
                    keys,
                    dormancies,
                    layer_props,
                    mask,
                )

                updates_carry_leaves = jax.tree_util.tree_leaves(updates_carry)
                updates = [
                    updates_carry_leaves[i]
                    for i in range(0, len(updates_carry_leaves), 3)
                ]

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

                return (next_opt_state_actor, next_opt_state_critic, updates_flat)

        return _Opt()
