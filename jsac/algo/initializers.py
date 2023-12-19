from jax import random
import optax
from jsac.algo.models import ActorModel, CriticModel, Temperature
from flax.training.train_state import TrainState
from flax.core.frozen_dict import freeze
from flax import traverse_util
from jsac.helpers.utils import MODE
import numpy as np
import jax.numpy as jnp 
import torch
import os


def get_init_data(init_image_shape, 
                  init_proprioception_shape, 
                  mode):
    init_image = None
    init_proprioception = None 

    if mode == MODE.IMG or mode == MODE.IMG_PROP:
        init_image = np.random.randint(
            0, 256, size=(1, *init_image_shape), dtype=np.uint8)
    if mode == MODE.PROP or mode == MODE.IMG_PROP:
        init_proprioception = np.random.uniform(
            size=(1, *init_proprioception_shape)).astype(np.float32)

    return init_image, init_proprioception
    

def init_inference_actor(rng, 
                         init_image_shape, 
                         init_proprioception_shape, 
                         action_dim, 
                         net_params, 
                         rad_offset,
                         mode=MODE.IMG_PROP):
    
    model = ActorModel(net_params,
                       action_dim,  
                       rad_offset, 
                       mode)
    
    init_image, init_proprioception = get_init_data(
        init_image_shape, 
        init_proprioception_shape, 
        mode)

    rng, *keys = random.split(rng, 5)
    model.init(keys[0], 
               keys[1:],
               init_image, 
               init_proprioception)['params']

    return rng, model

def init_actor(rng, 
               seed,
               critic, 
               learning_rate, 
               init_image_shape, 
               init_proprioception_shape, 
               action_dim, 
               net_params, 
               rad_offset,  
               use_critic_encoder=True, 
               mode=MODE.IMG_PROP):
    
    model = ActorModel(net_params,
                       action_dim,  
                       rad_offset, 
                       mode)

    rng, *keys = random.split(rng, 5)
    
    init_image, init_proprioception = get_init_data(
        init_image_shape, 
        init_proprioception_shape, 
        mode)
    
    params = model.init(keys[0], 
                        keys[1:],
                        init_image,
                        init_proprioception)['params']
    
    if use_critic_encoder:
        params['encoder'] = critic.params['encoder']

    tx = optax.adam(learning_rate=learning_rate)
    
    return rng, TrainState.create(apply_fn=model.apply, 
                                  params=params, 
                                  tx=tx)


def init_critic(rng,
                seed, 
                learning_rate, 
                init_image_shape, 
                init_proprioception_shape, 
                action_dim, 
                net_params, 
                rad_offset, 
                mode=MODE.IMG_PROP):

    model = CriticModel(net_params, 
                        action_dim, 
                        rad_offset,  
                        mode)
    
    rng, *keys = random.split(rng, 4)
    init_actions = random.uniform(keys[0], (1, action_dim))

    init_image, init_proprioception = get_init_data(
        init_image_shape, 
        init_proprioception_shape, 
        mode)
    
    params = model.init(keys[1], 
                        keys[2:],
                        init_image, 
                        init_proprioception, 
                        init_actions)['params']

    tx = optax.adam(learning_rate=learning_rate)

    return rng, TrainState.create(apply_fn=model.apply, 
                                  params=params, 
                                  tx=tx)


def init_temperature(rng, learning_rate, alpha=1.0):
    model = Temperature(initial_temperature=alpha)
    rng, key = random.split(rng)
    params = model.init(key)['params']

    tx = optax.adam(learning_rate=learning_rate)

    return rng, TrainState.create(apply_fn=model.apply, 
                                  params=params, 
                                  tx=tx)
