from jax import random
import optax
from jsac.algo.models import ActorModel, CriticModel, Temperature
from flax.training.train_state import TrainState
from flax.core.frozen_dict import freeze
from flax import traverse_util
from jsac.helpers.utils import MODE
from jsac.helpers.relod_models import ActorModel as RelodActorModel
from jsac.helpers.relod_models import CriticModel as RelodCriticModel
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

def set_torch_seed(seed):
    torch.set_printoptions(precision=10)
    torch.set_default_dtype(torch.float32)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def set_actor_weights(actor, params, config, mode):
    if mode == MODE.IMG or mode == MODE.IMG_PROP:
        num_conv_layers=len(config['conv'])
        for i in range(num_conv_layers):
            j = i * 2
            conv_w = actor.encoder.convs[j].weight.data.numpy().transpose(2, 3, 1, 0)
            params['encoder'][f'encoder_conv_{i}']['kernel'] = jnp.array(conv_w)
            
            conv_b = actor.encoder.convs[j].bias.data.numpy()
            params['encoder'][f'encoder_conv_{i}']['bias'] = jnp.array(conv_b)

    num_linear_layers=len(config['mlp'])
    for i in range(num_linear_layers):
        j = i * 2
        linear_w = actor.trunk[j].weight.data.numpy().transpose()
        params['MLP_0'][f'Dense_{i}']['kernel'] = jnp.array(linear_w)
        
        linear_b = actor.trunk[j].bias.data.numpy()
        params['MLP_0'][f'Dense_{i}']['bias'] = jnp.array(linear_b)

    j = num_linear_layers * 2
    linear_w = actor.trunk[j].weight.data.numpy().transpose()
    params[f'Dense_0']['kernel'] = jnp.array(linear_w)

    linear_b = actor.trunk[j].bias.data.numpy()
    params['Dense_0']['bias'] = jnp.array(linear_b)

    return params


def set_critic_weights(critic, params, config, mode):
    if mode == MODE.IMG or mode == MODE.IMG_PROP:
        num_conv_layers=len(config['conv'])
        for i in range(num_conv_layers):
            j = i * 2
            conv_w = critic.encoder.convs[j].weight.data.numpy().transpose(2, 3, 1, 0)
            params['encoder'][f'encoder_conv_{i}']['kernel'] = jnp.array(conv_w)
            
            conv_b = critic.encoder.convs[j].bias.data.numpy()
            params['encoder'][f'encoder_conv_{i}']['bias'] = jnp.array(conv_b)

    num_linear_layers=len(config['mlp']) + 1
    for i in range(num_linear_layers):
        j = i * 2
        linear_q1_w = critic.Q1.trunk[j].weight.data.numpy().transpose() 
        params['QFunction_0']['MLP_0'][f'Dense_{i}']['kernel'] = jnp.array(linear_q1_w)
        
        linear_q1_b = critic.Q1.trunk[j].bias.data.numpy()
        params['QFunction_0']['MLP_0'][f'Dense_{i}']['bias'] = jnp.array(linear_q1_b)

    num_linear_layers=len(config['mlp']) + 1
    for i in range(num_linear_layers):
        j = i * 2
        linear_q2_w = critic.Q2.trunk[j].weight.data.numpy().transpose() 
        params['QFunction_1']['MLP_0'][f'Dense_{i}']['kernel'] = jnp.array(linear_q2_w)
        
        linear_q2_b = critic.Q2.trunk[j].bias.data.numpy()
        params['QFunction_1']['MLP_0'][f'Dense_{i}']['bias'] = jnp.array(linear_q2_b)

    return params


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
    
    set_torch_seed(seed)
    h, w, c = init_image_shape
    relod_image_shape = (c, h, w)
    relod_actor = RelodActorModel(relod_image_shape,
                                  init_proprioception_shape,
                                  action_dim,
                                  net_params,
                                  rad_offset)

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
    

    params = set_actor_weights(relod_actor, params, net_params, mode)
    if use_critic_encoder:
        params['encoder'] = critic.params['encoder']
    del relod_actor

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
    
    set_torch_seed(seed)
    h, w, c = init_image_shape
    relod_image_shape = (c, h, w)
    relod_critic = RelodCriticModel(relod_image_shape, 
                                    init_proprioception_shape, 
                                    action_dim, 
                                    net_params, 
                                    rad_offset)
    

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
    
    params = set_critic_weights(relod_critic, params, net_params, mode)
    del relod_critic

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
