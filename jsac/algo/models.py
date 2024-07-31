from typing import Optional, Sequence, Any
 
import flax.linen as nn
import jax 
import jax.numpy as jnp
from jax import random, vmap 
import functools
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class MODE:
    IMG = 'img'
    IMG_PROP = 'img_prop'
    PROP = 'prop'


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale, dtype=jnp.float32)


@functools.partial(jax.jit, static_argnames=('image_shape'))
def augment(image, start_h, start_w, image_shape):
    return jax.lax.dynamic_slice(image, 
                                 (start_h, start_w, 0), 
                                 image_shape)


class SpatialSoftmax(nn.Module):
    height: float
    width: float
    channel: float

    def setup(self):
      pos_x, pos_y = jnp.meshgrid(
         jnp.linspace(-1., 1., self.height, dtype=jnp.float32),
         jnp.linspace(-1., 1., self.width, dtype=jnp.float32)
      )
      self._pos_x = pos_x.reshape(self.height*self.width)
      self._pos_y = pos_y.reshape(self.height*self.width)

    @nn.compact
    def __call__(self, feature):  
        feature = feature.transpose(0, 3, 1, 2)
        feature = feature.reshape(-1, self.height*self.width)
    
        softmax_attention = nn.activation.softmax(feature, axis = -1)

        expected_x = jnp.sum(self._pos_x*softmax_attention, axis = 1, 
                             keepdims=True)
        expected_y = jnp.sum(self._pos_y*softmax_attention, axis = 1,
                             keepdims=True)

        expected_xy = jnp.concatenate(axis = 1, arrays=(expected_x, expected_y))
        
        feature_keypoints = expected_xy.reshape(-1, self.channel * 2) 
        
        return feature_keypoints
    

class Encoder(nn.Module):
    net_params: dict 
    spatial_softmax: bool = True
    rad_offset: float = 0.01
    mode: str = MODE.IMG_PROP
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, 
                 keys,
                 images, 
                 proprioceptions, 
                 apply_rad=False,
                 stop_gradient=False):          
        
        if self.mode == MODE.PROP:
            return proprioceptions
        
        conv_params = self.net_params['conv']
        
        batch_size, height, width, channel = images.shape

        rad_h = max(round(self.rad_offset * height) * 2, 1)
        rad_w = max(round(self.rad_offset * width) * 2, 1)
        rad_image_shape = ((height - (2 * rad_h)), 
                           (width - (2 * rad_w)), 
                           channel)   
        get_augments = vmap(augment, in_axes=(0, 0, 0, None))
            
        if not apply_rad:
            # Still need to crop the images
            crop_height = jnp.ones((batch_size,), dtype=jnp.int32) * rad_h
            crop_width = jnp.ones((batch_size,), dtype=jnp.int32) * rad_w
        else:
            crop_height = random.randint(keys[0], (batch_size,), 0, rad_h+1, jnp.int32)
            crop_width = random.randint(keys[1], (batch_size,), 0, rad_w+1, jnp.int32)

        images = get_augments(images,
                              crop_height,
                              crop_width,
                              rad_image_shape)

        x = images.astype(self.dtype)
        x = (x / 255.0) - 0.5

        for i, (_, out_channel, kernel_size, stride) in enumerate(conv_params):
            layer_name = 'encoder_conv_' + str(i)

            x = nn.Conv(features=out_channel, 
                        kernel_size=(kernel_size, kernel_size),
                        strides=stride,
                        padding=0,  
                        kernel_init=nn.initializers
                        .delta_orthogonal(dtype=jnp.float32), 
                        name=layer_name 
            )(x)

            if i < len(conv_params) - 1:
                x = nn.relu(x)

        b, height, width, channel = x.shape
        
        if self.spatial_softmax:
            x = SpatialSoftmax(width, height, channel, 
                                name='encoder_spatialsoftmax')(x)
        else:
            x = jnp.reshape(x, (b, -1)) 

        if stop_gradient:
            x = jax.lax.stop_gradient(x)

        if self.mode == MODE.IMG_PROP:
           x = jnp.concatenate(axis = -1, arrays=(x, proprioceptions)) 

        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activate_final: int = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


class ActorModel(nn.Module):
    net_params: dict 
    action_dim: int
    rad_offset: float = 0.01
    spatial_softmax: bool = True
    mode: str = MODE.IMG_PROP
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, 
                 keys, 
                 images, 
                 proprioceptions, 
                 log_std_min=-10,
                 log_std_max=10,
                 apply_rad=False):

        latents = Encoder(self.net_params, 
                          self.spatial_softmax,
                          self.rad_offset,
                          self.mode,
                          self.dtype,
                          name='encoder')(keys[1:],
                                          images, 
                                          proprioceptions, 
                                          apply_rad,
                                          True)
        
        outputs = MLP(self.net_params['mlp'], activate_final=True, dtype=self.dtype)(latents)
        init = nn.initializers.zeros_init()
        mu = nn.Dense(self.action_dim, kernel_init=init, dtype=self.dtype)(outputs)
        log_std = nn.Dense(self.action_dim, kernel_init=init, dtype=self.dtype)(outputs)
        log_std = jnp.clip(log_std, log_std_min, log_std_max)

        ## From https://github.com/ikostrikov/jaxrl
        mu = nn.tanh(mu)
        base_dist = tfd.MultivariateNormalDiag(loc=mu,
                                               scale_diag=jnp.exp(log_std))

        dist = tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        pi = dist.sample(seed=keys[0])
        log_pi = dist.log_prob(pi)
        
        return mu, pi, log_pi, log_std
    
    def __hash__(self): 
        return id(self)


class QFunction(nn.Module):
    hidden_dims: Sequence[int]
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, latents, actions):
        inputs = jnp.concatenate([latents, actions], -1)
        critic = MLP((*self.hidden_dims, 1))(inputs)
        return jnp.squeeze(critic, -1)


class CriticModel(nn.Module):
    net_params: dict  
    action_dim: int
    rad_offset: float = 0.01
    spatial_softmax: bool = True
    mode: str = MODE.IMG_PROP
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, 
                 keys,
                 images, 
                 proprioceptions, 
                 actions,  
                 apply_rad=False,
                 stop_gradient=False):
        
        latents = Encoder(self.net_params, 
                          self.spatial_softmax,
                          self.rad_offset,
                          self.mode,
                          self.dtype,
                          name='encoder')(keys,
                                          images, 
                                          proprioceptions, 
                                          apply_rad,
                                          stop_gradient)

        VmapCritic = nn.vmap(
            QFunction,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        qs = VmapCritic(self.net_params['mlp'])(latents, actions)
        
        return qs 
    

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            'log_temp', 
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature), dtype=jnp.float32))
        return jnp.exp(log_temp)