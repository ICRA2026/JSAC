## Non-visual Tasks: Mujoco Environments

### Trainable parameters:

| Network Name | Trainable Parameters |
|----------|----------|
| Critic | 5,329,925 |
| Actor | 1,068,038 |
| Temperature | 1 |
| **Total** | **6,397,964** |

### Hyperparameters:
| Name                      | Default Value   | Description                                                                                                                               |
|---------------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `mode`                    | `prop`        | Proprioception only mode.                                                     |
| `replay_buffer_capacity`  | `1000000`       | Size of the replay buffer.                                                                                                        |
| `init_steps`              | `5000`          | Number of initial steps before training starts.                                                             |
| `env_steps`               | `1000000`       | Total number of steps.                                                                     |
| `batch_size`              | `256`           | Batch size for training network updates.                                                                                                  |
| `sync_mode`               | `True`          | If `True`, network updates occur sequentially after env steps. If `False` (async), updates happen in parallel.                                |
| `global_norm`             | `1.0`           | Value for `optax.clip_by_global_norm()` applied to actor and critic network gradients.                                                    |
| `layer_norm`              | `False`         | If `True`, apply `nn.LayerNorm` after each hidden layer in networks.                                                                        |
| `apply_weight_clip`       | `False`         | If `True`, applies weight clipping to network weights based on [arXiv:2407.01704](https://arxiv.org/abs/2407.01704).                           |
| `critic_lr`               | `3e-4`          | Learning rate for the critic network(s).                                                                                                  |
| `num_critic_networks`     | `5`             | Number of critic networks to use.                                                                                |
| `num_critic_updates`      | `1`             | Number of critic network updates to perform per environment step.                                                                           |
| `critic_tau`              | `0.005`         | Interpolation factor for Polyak averaging (soft updates) of the target critic networks.                                                     |
| `critic_target_update_freq` | `1`             | Frequency (in environment steps) at which to update the target critic networks.                                                           |
| `actor_lr`                | `3e-4`          | Learning rate for the actor network.                                                                                                      |
| `actor_update_freq`       | `1`             | Frequency (in environment steps) at which to update the actor network.                                                                    |
| `temp_lr`                 | `3e-4`          | Learning rate for the SAC temperature parameter (alpha).                                                                                  |
| `init_temperature`        | `1.0`           | Initial value for the SAC temperature parameter (alpha).                                                                                  |
| `discount`                | `0.99`          | Discount factor (gamma) for future rewards.                                                                                               |



## Image-only Tasks: DeepMind Control (DMC) Environments

### Trainable parameters:

| Network Name | Trainable Parameters |
|----------|----------|
| Encoder | 88,288 |
| Critic | 6,800,581 |
| Actor | 2,312,396 |
| Temperature | 1 |
| **Total** | **9,201,266** |

### Hyperparameters:
| Name                      | Default Value   | Description                                                                                                                               |
|---------------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `mode`                    | `img`         | Image only observation          |
| `image_height`            | `96`            | Height of the input image observation (in pixels).                                                               |
| `image_width`             | `96`            | Width of the input image observation (in pixels).                                                                |
| `image_history`           | `3`             | Number of chronological image frames to stack.                                                 |
| `action_repeat`           | `2`             | Number of times to repeat each action in the environment.                                                                                 |
| `replay_buffer_capacity`  | `500000`        |                                                                                                              |
| `init_steps`              | `5000`          |                                                                                          |
| `env_steps`               | `500000`        |                                                                                                                  |
| `batch_size`              | `256`           |                                                                                                  |
| `sync_mode`               | `True`          |                                |
| `global_norm`             | `1.0`           |                                                   |
| `layer_norm`              | `False`         |                                                                        |
| `apply_weight_clip`       | `False`         |                           |
| `critic_lr`               | `3e-4`          |                                                                                                 |
| `num_critic_networks`     | `5`             |                                                                                                       |
| `num_critic_updates`      | `1`             |                                                                            |
| `critic_tau`              | `0.005`         |                                                     |
| `critic_target_update_freq` | `1`             |                                                        |
| `actor_lr`                | `3e-4`          |                                                                                                      |
| `actor_update_freq`       | `1`             |                                                                    |
| `spatial_softmax`         | `False`         | If `True`, use Spatial Softmax layer in the image encoder.                                           |
| `temp_lr`                 | `3e-4`          |                                                                                |
| `init_temperature`        | `0.1`           |                                                                              |
| `discount`                | `0.99`          |                                                                                           |


## Create2-Orin Reacher Task

### Trainable parameters:

| Network Name | Trainable Parameters |
|----------|----------|
| Encoder | 87,744 |
| Critic | 6,140,101 |
| Actor | 1,575,118 |
| Temperature | 1 |
| **Total** | **7,802,964** |

### Hyperparameters:
| Name                      | Default Value   | Description                                                                                                                               |
|---------------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `image_height`            | `60`                              | |
| `image_width`             | `80`                              | |
| `image_history`           | `2`                               | |
| `mode`                    | `'img_prop'`                      | |
| `apply_weight_clip`       | `True`                            | |
| `episode_length_time`     | `12.0`                            | |
| `dt`                      | `0.06`                            | Action sampling time (60ms). |
| `min_target_size`         | `0.40`                            | Target size threshold in the image. |
| `replay_buffer_capacity`  | `75000`                           | |
| `init_steps`              | `5000`                            | |
| `env_steps`               | `75000`                           | |
| `batch_size`              | `160`                             | |
| `sync_mode`               | `False`                           | |
| `global_norm`             | `1.0`                             | |
| `layer_norm`              | `True`                            | |
| `apply_weight_clip`       | `True`                             | Weight clipping value: `2.0 * √2.0`                           |
| `critic_lr`               | `2e-4`                            | |
| `num_critic_networks`     | `5`                               | |
| `num_critic_updates`      | `1`                               | |
| `critic_tau`              | `0.005`                           | |
| `critic_target_update_freq` | `1`                               | |
| `actor_lr`                | `2e-4`                            | |
| `actor_update_freq`       | `1`                               | |
| `actor_sync_freq`         | `16`                              | The updated actor network parameters are synchronized with the environment process every `actor_sync_freq` update steps. |
| `spatial_softmax`         | `False`                           | |
| `temp_lr`                 | `2e-4`                            | |
| `init_temperature`        | `0.1`                             | |
| `discount`                | `0.99`                            | |
