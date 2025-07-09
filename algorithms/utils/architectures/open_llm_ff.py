import warnings
from typing import Any, Optional

import flax
import gin
import jax
import jax.numpy as jnp

import algorithms.utils.base as opt_base
from algorithms.utils.learned_optimization.learned_optimization import tree_utils
from algorithms.utils.learned_optimization.learned_optimization.learned_optimizers import (
    common,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

PRNGKey = jnp.ndarray


def iter_proportion(iterations, total_its=100000):
    f32 = jnp.float32

    return iterations / f32(total_its)


@flax.struct.dataclass
class LLMOptState:
    params: Any
    rolling_features: common.MomAccumulator
    iteration: jnp.ndarray
    state: Any


@gin.configurable
class LLM_Open_FF:
    def __init__(self, function=None, lr=1e-2):

        self.func = function
        self.lr = lr

    def opt_fn(self) -> opt_base.Optimizer:
        # ALL MOMENTUM TIMESCALES
        decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])
        func = self.func
        lr_ = self.lr

        class _Opt(opt_base.Optimizer):
            def init(
                self,
                params: opt_base.Params,
                model_state: Any = None,
                num_steps: Optional[int] = None,
                key: Optional[PRNGKey] = None,
            ) -> LLMOptState:
                """Initialize opt state."""

                param_tree = jax.tree_util.tree_structure(params)

                keys = jax.random.split(key, param_tree.num_leaves)
                keys = jax.tree_util.tree_unflatten(param_tree, keys)

                return LLMOptState(
                    params=params,
                    state=model_state,
                    rolling_features=common.vec_rolling_mom(decays).init(params),
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                )

            def update(
                self,
                opt_state_actor: LLMOptState,
                crit_opt_state: LLMOptState,
                grad: Any,
                activations: float,
                key: Optional[PRNGKey] = None,
                training_prop=0,
                batch_prop=0,
                config=None,
                layer_props=None,
                model_state: Any = None,
                mask=None,
                opt_params=None,
                meta_network=None,
            ) -> LLMOptState:

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

                lr = lr_ * (1 - training_step_feature)

                def _update_tensor(p, g, mom, k, dorm, layer_prop, mask):

                    if not p.shape:
                        p = jnp.expand_dims(p, 0)
                        g = jnp.expand_dims(g, 0)

                        mom = jnp.expand_dims(mom, 0)
                        did_reshape = True
                    else:
                        did_reshape = False

                    rng, rng_ = jax.random.split(k)

                    rand = jax.random.normal(rng_, g.shape) * mask

                    update = func(
                        p,
                        mom[..., 0],
                        mom[..., 1],
                        mom[..., 2],
                        mom[..., 3],
                        mom[..., 4],
                        mom[..., 5],
                        layer_prop,
                        batch_prop,
                        training_prop,
                        dorm,
                        g,
                        rand,
                        lr,
                    )

                    update = update.reshape(p.shape)
                    return (update, update)

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
                    for i in range(0, len(updates_carry_leaves), 2)
                ]
                determ_update = [
                    updates_carry_leaves[i + 1]
                    for i in range(0, len(updates_carry_leaves), 2)
                ]

                updates = jax.tree_util.tree_unflatten(param_tree, updates)
                determ_update = jax.tree_util.tree_unflatten(param_tree, determ_update)
                # Make update globally 0
                updates_flat = jax.flatten_util.ravel_pytree(updates)[0]
                update_mean = updates_flat.mean()
                update_mean = jax.tree_util.tree_unflatten(
                    param_tree, jnp.tile(update_mean, param_tree.num_leaves)
                )

                def param_update(p, update):

                    new_param = p - update

                    return new_param

                next_params = jax.tree_util.tree_map(param_update, full_params, updates)

                # For simplicity, maitain different opt states between the actor and the critic
                next_opt_state_actor = LLMOptState(
                    params=tree_utils.match_type(
                        next_params["actor"], opt_state_actor.params
                    ),
                    rolling_features=tree_utils.match_type(
                        next_rolling_features_actor, opt_state_actor.rolling_features
                    ),
                    iteration=opt_state_actor.iteration + 1,
                    state=model_state,
                )

                next_opt_state_critic = LLMOptState(
                    params=tree_utils.match_type(
                        next_params["critic"], crit_opt_state.params
                    ),
                    rolling_features=tree_utils.match_type(
                        next_rolling_features_critic, crit_opt_state.rolling_features
                    ),
                    iteration=opt_state_actor.iteration + 1,
                    state=model_state,
                )

                return (next_opt_state_actor, next_opt_state_critic, updates_flat)

        return _Opt()
