# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import gymnasium as gym
import time
import logging
import numpy as np
import random
import threading
import jsac.envs.create2_orin_visual_reacher.senseact_create_env.create2_config as create2_config
from jsac.envs.create2_orin_visual_reacher.senseact_create_env import utils as utils

from multiprocessing import Array, Value
from collections import deque

from jsac.envs.create2_orin_visual_reacher.senseact_create_env.rtrl_base_env import RTRLBaseEnv 
from jsac.envs.create2_orin_visual_reacher.senseact_create_env.create2_communicator import Create2Communicator
from jsac.envs.create2_orin_visual_reacher.senseact_create_env.create2_observation import Create2ObservationFactory
from jsac.envs.create2_orin_visual_reacher.senseact_create_env.sharedbuffer import SharedBuffer
from jsac.envs.create2_orin_visual_reacher.depstech_camera_communicator import CameraCommunicator


OB_TYPE_1 = "MASK"
OB_TYPE_2 = "OH"
OB_TYPE_3 = "MASK_OH"

def print_with_delay(text, delay=0.2):
    threading.Timer(delay, print, args=(text,)).start()

class Create2VisualReacherEnv(RTRLBaseEnv, gym.Env):
    """Create2 environment for training it drive forward.
    By default this environment observes the infrared character (binary state for detecting the dock),
    bump sensor, and last action.
    The reward is the forward reward.  Episode ends early when the bump sensor is triggered.
    TODO:
        * move all common methods between this class and docking_env to a base class
    """

    def __init__(self, episode_length_time=30, port='/dev/ttyUSB0', obs_history=1, dt=0.015, image_shape=(0, 0, 0),
                 camera_id=0, min_target_size=0.5, pause_before_reset=0, pause_after_reset=0, dense_reward=False, 
                 multi_target=False, ob_type=OB_TYPE_1, **kwargs):
        """Constructor of the environment.
        Args:
            episode_length_time: A float duration of an episode defined in seconds
            port:                the serial port to the Create2 (eg. '/dev/ttyUSB0')
            obs_history:         the number of observation history to keep
            dt:                  the cycle time in seconds
            auto_unwind:         boolean of whether we want to execute the auto cable-unwind code
            rllab_box:           whether we are using rllab algorithm or not
            **kwargs:            the remaining arguments passed to the base class
        """
        self._obs_history = obs_history
        self._episode_step_ = Value('i', 0)
        self._episode_length_time = episode_length_time
        self._episode_length_step = int(episode_length_time / dt)
        self.pause_before_reset = pause_before_reset
        self.pause_after_reset = pause_after_reset
        self._internal_timing = 0.015
        self._ob_type = ob_type

        if multi_target:
            self._multi_target = True
            self._target1 = 0 # 'Pink'
            self._target2 = 1 # 'Green'
            # self._target1_color_bgr = (204, 102, 255)
            # self._target2_color_bgr = (0, 255, 0) 
            self._target1_hsv = ((134, 71, 193), (191, 219, 255))
            self._target2_hsv = ((40, 40, 120), (75, 100, 255))
            self._current_target = self._target1
            self._target_oh = [[1.0, 0.0], [0.0, 1.0]]
        else:
            self._hsv_mask = ((134, 71, 193), (191, 219, 255))   ## PINK
        self._min_target_size = min_target_size
        self._min_battery = 1000
        self._max_battery = 1700
        self._dense_reward = dense_reward

        # get the opcode for our main action (only 1 action)
        self._main_op = 'drive_direct'
        self._extra_ops = ['safe', 'seek_dock', 'drive']
        main_opcode = create2_config.OPCODE_NAME_TO_CODE[self._main_op]
        extra_opcodes = [create2_config.OPCODE_NAME_TO_CODE[op] for op in self._extra_ops]

        # store the previous action/reward to be shared across processes
        self._prev_action_buffer = deque([], maxlen=15)
        self._prev_action_ = np.frombuffer(Array('i', 30).get_obj(), dtype='i')

        # create factory with common arguments for making an observation dimension
        observation_factory = Create2ObservationFactory(main_op=self._main_op,
                                                        dt=dt,
                                                        obs_history=self._obs_history,
                                                        internal_timing=self._internal_timing,
                                                        prev_action=self._prev_action_)

        # the definition of the observed state and the associated custom modification (if any)
        # before passing to the learning algorithm
        self._observation_def = [
            # observation_factory.make_dim('light bump left signal'),
            # observation_factory.make_dim('light bump front left signal'),
            # observation_factory.make_dim('light bump center left signal'),
            # observation_factory.make_dim('light bump center right signal'),
            # observation_factory.make_dim('light bump front right signal'),
            # observation_factory.make_dim('light bump right signal'),
            observation_factory.make_dim('previous action')
        ]

        # extra packets we need for proper reset and charging
        self._extra_sensor_packets = ['bumps and wheel drops', 'battery charge',
                                      'oi mode', 'distance','charging sources available',
                                      'cliff left', 'cliff front left', 'cliff front right', 'cliff right']
        main_sensor_packet_ids = [d.packet_id for d in self._observation_def if d.packet_id is not None]
        extra_sensor_packet_ids = [create2_config.PACKET_NAME_TO_ID[nm] for nm in self._extra_sensor_packets]

        # TODO: move this out to some base class?
        from gymnasium.spaces import Box as GymBox  # use this for baselines algos
        Box = GymBox

        # go thru the main opcode (just direct_drive in this case) and add the range of each param
        # XXX should the action space include the opcode? what about op that doesn't have parameters?
        self._action_space = Box(
            low=np.array([r[0] for r in create2_config.OPCODE_INFO[main_opcode]['params'].values()]),
            high=np.array([r[1] for r in create2_config.OPCODE_INFO[main_opcode]['params'].values()])
        )

        # loop thru the observation dimension and get the lows and highs
        low=np.concatenate([d.lows for d in self._observation_def])
        high=np.concatenate([d.highs for d in self._observation_def])

        if self._ob_type == OB_TYPE_2 or self._ob_type == OB_TYPE_3:
            low = np.concatenate([low, [0.0, 0.0]])
            high = np.concatenate([high, [1.0, 1.0]])

        if self._ob_type == OB_TYPE_1 or self._ob_type == OB_TYPE_3:
            self._num_img_channels = 4
        else:
            self._num_img_channels = 3

        self._observation_space = Box(
            low=low,
            high=high
        )

        # self._comm_name = 'Create2'
        communicator_setups = {}
        buffer_len = int(dt / self._internal_timing + 1)
        communicator_setups['Create2'] = {'Communicator': Create2Communicator,
                                                 # have to read in this number of packets everytime to support
                                                 # all operations
                                                 'num_sensor_packets': buffer_len,
                                                 'kwargs': {'sensor_packet_ids': main_sensor_packet_ids +
                                                                                 extra_sensor_packet_ids,
                                                            'opcodes': [main_opcode] + extra_opcodes,
                                                            'port': port,
                                                            'buffer_len': 2 * buffer_len,
                                                           }
                                            }
        

        self._roomba_obs_buffer = SharedBuffer(
                buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
                array_len=len(self._observation_space.low)+2,
                array_type='d',
                np_array_type='d',
                )

        if image_shape != (0, 0, 0):
            image_stack = int(image_shape[-1] // self._num_img_channels)
            communicator_setups['Camera'] = {'Communicator': CameraCommunicator,
                                             'num_sensor_packets': image_stack,
                                             'kwargs': 
                                                    {'device_id': camera_id,
                                                     'res': (image_shape[1], image_shape[0], self._num_img_channels) # communicator uses w, h
                                                    }
                                            }

            self._image_obs_buffer = SharedBuffer(
                buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
                array_len=image_shape[0]*image_shape[1]*image_shape[2]+1,
                array_type='B',
                np_array_type='B',
                )
            self._image_reward_buffer = SharedBuffer(
                buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
                array_len=1,
                array_type='d',
                np_array_type='d',
                )
        
            self._image_buffer = deque([], maxlen=image_stack)
            self._blank_img = np.zeros((image_shape[0],image_shape[1],self._num_img_channels))

        self._image_shape = image_shape
        self._image_space = Box(low=0, high=255, shape=self._image_shape)
        super().__init__(communicator_setups=communicator_setups,
                        action_dim=len(self._action_space.low),
                        observation_dim=-2, # dont use the base class sensation buffer
                        dt=dt,
                        **kwargs)
    
    def _sensor_to_sensation_(self):
        # overwrite this to support image
        for name, comm in self._sensor_comms.items():
            if comm.sensor_buffer.updated():
                sensor_window, timestamp_window, index_window = comm.sensor_buffer.read_update(self._num_sensor_packets[name])
                if name == 'Create2':
                    s = self._compute_roomba_obs_(sensor_window, timestamp_window, index_window)
                    self._roomba_obs_buffer.write(s, timestamp=timestamp_window[-1])
                elif name == 'Camera':
                    s, r = self._compute_image_obs_(sensor_window, timestamp_window, index_window)
                    self._image_obs_buffer.write(s, timestamp=timestamp_window[-1])
                    self._image_reward_buffer.write(r, timestamp=timestamp_window[-1])
                else:
                    raise NotImplementedError('Unsupported communicator')

    def _read_sensation(self):
        # overwrite this to support image
        roomba_obs_r_d, roomba_obs_timestamp, _ = self._roomba_obs_buffer.read_update()
        roomba_obs, r_r, r_d = roomba_obs_r_d[0][:-2], roomba_obs_r_d[0][-2], roomba_obs_r_d[0][-1]
        image = None
        im_r = 0
        im_d = 0
        if self._image_shape != (0, 0, 0):
            image_d, image_timestamp, _ = self._image_obs_buffer.read_update()
            while True:
                delay = abs(image_timestamp[-1] - roomba_obs_timestamp[-1])
                if delay < 1.0:
                    break
                image_d, image_timestamp, _ = self._image_obs_buffer.read_update()
                roomba_obs_r_d, roomba_obs_timestamp, _ = self._roomba_obs_buffer.read_update()
                roomba_obs, r_r, r_d = roomba_obs_r_d[0][:-2], roomba_obs_r_d[0][-2], roomba_obs_r_d[0][-1]

            image, im_d = image_d[0][:-1], image_d[0][-1]
            im_r, _, _ = self._image_reward_buffer.read_update()
            im_r = im_r[0][0]

            delay = abs(image_timestamp[-1] - roomba_obs_timestamp[-1])
            if delay > self._dt:
                print('Warning: image time and proprioception time is different by: {}s.'.format(delay))
                #print
            # unflatten image
            stacks = int(self._image_shape[-1]//self._num_img_channels)
            height = self._image_shape[0]
            width = self._image_shape[1]
            image = image.reshape((stacks, height, width, self._num_img_channels))
            # image = np.transpose(image.reshape((stacks, height, width, 3)), (0, 3, 1, 2)) # s, c, h, w
            image = np.concatenate(image, axis=-1) # change to self._image_shape

        #print('r_r:', r_r, "im_r:", im_r)
        done = (r_d or im_d)
        
        reward = r_r+im_r
        
        if self._dense_reward:
            reward = im_r
            if done > 0.5:
                reward = 5.0
        
        sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()
        battery_charge =  sensor_window[-1][0]['battery charge']

        # if self._multi_target:
        #     rect_h, rect_w = height//8, width//4
        #     top = height - rect_h
        #     left = (width - rect_w) // 2
        #     right = left + rect_w
        #     color = self._target1_color_bgr if self._current_target == self._target1 else self._target2_color_bgr
        #     for i in range(stacks):
        #         image[top:height, left:right, 3*i:3*i+3] = color

        return (image, roomba_obs), reward,  done, {'battery_charge': battery_charge}

    def _compute_image_obs_(self, sensor_window, timestamp_window, index_window):
        reward, done = self._calc_image_reward(sensor_window)

        if len(self._image_buffer) < self._image_buffer.maxlen:
            for _ in range(self._image_buffer.maxlen - len(self._image_buffer)):
                self._image_buffer.append(self._blank_img)
        flattened = np.concatenate(self._image_buffer, axis=-1).flatten()
        return np.concatenate((flattened, [done])).astype('uint8'), reward

    def _compute_roomba_obs_(self, sensor_window, timestamp_window, index_window):
        """The required _computer_sensation_ interface.
        Args:
            name:               the name of communicator the sense is from
            sensor_window:      an array of size num_sensor_packets each containing 1 complete observation packets
            timestamp_window:   array of timestamp corresponds to the sensor_window
            index_window:       array of count corresponds to the sensor_window
        Returns:
            A numpy array with [:-2] the sensation, [-2] the reward, [-1] the done flag
        """
        # construct the actual sensation

        actual_obs = []
        for d in self._observation_def:
            res = d.normalized_handler(sensor_window)
            actual_obs.extend(res)
        
        if self._ob_type == OB_TYPE_2 or self._ob_type == OB_TYPE_3:
            actual_obs.extend(self._target_oh[self._current_target])

        # accumulate the rotation information
        # self._total_rotation += sensor_window[-1][0]['angle']

        reward, done = self._calc_roomba_obs_reward(sensor_window)

        return np.concatenate((actual_obs, [reward], [done]))

    def _compute_actuation_(self, action, timestamp, index):
        """The required _compute_actuator_ interface.
        The side effect is to write the output to self._actuation_packet_[name] with [opcode, *action]
        Args:
            action:      an array of 2 numbers correspond to the speed of the left & right wheel
            timestamp:   the timestamp when the action was written to buffer
            index:       the action count
        """
        # add a safety check for any action with nan or inf
        if any([not np.isfinite(a) for a in action]):
            logging.warning("Invalid action received: {}".format(action))
            return

        # pass int only action

        action = action.astype('i')

        self._actuation_packet_['Create2'] = np.concatenate(
            ([create2_config.OPCODE_NAME_TO_CODE[self._main_op]], action))
        
        self._prev_action_buffer.append(action)
        prev_actions = np.concatenate(self._prev_action_buffer, axis=-1).flatten()
        np.copyto(self._prev_action_, prev_actions)

    def _reset_(self):
        """The required _reset_ interface.
        This method does the handling of charging the Create2, repositioning, and set to the correct mode.
        """
        
        if self._multi_target:
            if random.random() < 0.5:
                self._current_target = self._target2 if self._current_target == self._target1 else self._target1
            else:
                self._current_target = random.choice([self._target1, self._target2])

        logging.info("Resetting...")
        self._write_opcode('drive', 0, 0)
        time.sleep(0.1)
        # N.B: pause_before_reset should be greater than zero only for demo purposes
        time.sleep(self.pause_before_reset)

        self._episode_step_.value = -1
        for i in range(15):
            self._prev_action_buffer.append(np.array([0, 0]))
        
        prev_actions = np.concatenate(self._prev_action_buffer, axis=-1).flatten()
        np.copyto(self._prev_action_, prev_actions)

        for d in self._observation_def:
            d.reset()

        # wait for create2 to startup properly if just started (ie. wait to actually start receiving observation)
        while not self._sensor_comms['Create2'].sensor_buffer.updated():
            time.sleep(0.01)

        sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()
        print('current charge:', sensor_window[-1][0]['battery charge'])
        if sensor_window[-1][0]['battery charge'] <= self._min_battery:
            print("Waiting for Create2 to be docked.")
            if sensor_window[-1][0]['charging sources available'] <= 0:
                self._write_opcode('drive_direct', 0, 0)
                time.sleep(0.75)
                self._write_opcode('seek_dock')
                time.sleep(10)
            self._wait_until_charged()
            sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()

        # Always switch to SAFE mode to run an episode, so that Create2 will switch to PASSIVE on the
        # charger.  If the create2 is in any other mode on the charger, we will not be able to detect
        # the non-responsive sleep mode that happens at the 60 seconds mark.
        logging.info("Setting Create2 into safe mode.")
        self._write_opcode('safe')
        time.sleep(0.1)

        # after charging/docked, try to drive away from the dock if still on it
        if sensor_window[-1][0]['charging sources available'] > 0:
            logging.info("Undocking the Create2.")
            self._write_opcode('drive_direct', -250, -250)
            time.sleep(0.75)
            self._write_opcode('drive_direct', 0, 0)
            time.sleep(0.1)

        # rotate and drive backward 
        logging.info("Moving Create2 into position.")
        target_values = [400, 400]
        # target_values = [-300, -300]
        move_time_1 = np.random.uniform(low=1, high=1.5)
        move_time_2 = np.random.uniform(low=0.3, high=0.7)
        rotate_time_1 = np.random.uniform(low=0.25, high=0.75)
        rotate_time_2 = np.random.uniform(low=0.5, high=1)
        direction = np.random.choice((1, -1))
        
        # rotate
        self._write_opcode('drive_direct', *(400*direction, -400*direction))
        time.sleep(rotate_time_1)
        self._write_opcode('drive', 0, 0)
        time.sleep(0.1)

        # back
        self._write_opcode('drive_direct', *target_values)
        time.sleep(move_time_1)
        self._write_opcode('drive', 0, 0)
        time.sleep(0.1)

        # rotate
        direction = np.random.choice((1, -1))
        self._write_opcode('drive_direct', *(400*direction, -400*direction))
        time.sleep(rotate_time_2)
        self._write_opcode('drive', 0, 0)
        time.sleep(0.1)
        
        # back
        self._write_opcode('drive_direct', *[-400, -400])
        # self._write_opcode('drive_direct', *[300, 300])
        time.sleep(move_time_2)
        self._write_opcode('drive', 0, 0)
        time.sleep(0.1)
        '''
        rand_state_array_type, rand_state_array_size, rand_state_array = utils.get_random_state_array(
            self._rand_obj_.get_state()
        )
        np.copyto(self._shared_rstate_array_, np.frombuffer(rand_state_array, dtype=rand_state_array_type))
        '''

        # make sure in SAFE mode in case the random drive caused switch to PASSIVE, or
        # create2 stuck somewhere and require human reset (don't want an episode to start
        # until fixed, otherwise we get a whole bunch of one step episodes)
        sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()
        while sensor_window[-1][0]['oi mode'] != 2:
            logging.warning("Create2 not in SAFE mode, reattempting... (might require human intervention).")
            self._write_opcode('full')
            time.sleep(0.2)
            self._write_opcode('drive_direct', -50, -50)
            time.sleep(0.5)
            self._write_opcode('drive', 0, 0)
            time.sleep(0.1)
            self._write_opcode('safe')
            time.sleep(0.2)

            sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()

        # N.B: pause_after_reset should be greater than zero only for demo purposes
        time.sleep(self.pause_after_reset)

        # don't want the state during reset pollute the first sensation
        time.sleep(2 * self._internal_timing)

        # wait for camera to startup properly
        if self._image_shape != (0, 0, 0):
            # while not self._sensor_comms['Camera'].sensor_buffer.updated():
            #     time.sleep(0.01)
            while not self._sensor_comms['Camera'].sensor_buffer.updated():
                time.sleep(0.01)

            # for _ in range(SharedBuffer.DEFAULT_BUFFER_LEN):
            #     _, _, _ = self._image_obs_buffer.read_update()


        for _ in range(self._image_buffer.maxlen):
            self._image_buffer.append(self._blank_img)

        print("Reset completed.")
        if self._multi_target:
            if self._current_target == 0:
                print_with_delay('Target: Pink')
            else:
                print_with_delay('Target: Green') 

    def _check_done(self, env_done):
        """The required _check_done_ interface.
        Args:
            env_done:   whether the environment is done from _compute_sensation_
        Returns:
            A boolean flag for done
        """
        self._episode_step_.value += 1
        # change
        # return self._episode_step_.value >= self._episode_length_step or env_done
        return env_done
        
    def _calc_roomba_obs_reward(self, sensor_window):
        """Helper to calculate reward.
        Args:
            sensor_window: the sensor_window from _compute_sensation_
        Returns:
            A tuple of (reward, done)
        """
        
        bw = 0
        for p in range(int(self._dt / self._internal_timing)):
            bw |= sensor_window[-1 - p][0]['bumps and wheel drops']
        
        cl = 0
        for p in range(int(self._dt / self._internal_timing)):
            cl += sensor_window[-1 - p][0]['cliff left']
            cl += sensor_window[-1 - p][0]['cliff front left']
            cl += sensor_window[-1 - p][0]['cliff front right']
            cl += sensor_window[-1 - p][0]['cliff right']
        
        reward = 0

        charging_sources_available = sensor_window[-1][0]['charging sources available']

        oi_mode = sensor_window[-1][0]['oi mode']
        if oi_mode == 1 and charging_sources_available == 0 and cl == 0:
            self._write_opcode('safe')

        # If wheel dropped, it's done and result in a big penalty.
        done = 0
        if (bw >> 2) > 0:
            done = 1
            reward = -self._episode_length_step

        return reward, done

    def _calc_image_reward(self, sensor_window):
        reward = 0.0
        done = 0
        image = sensor_window[-1]
        image = image.reshape(self._image_shape[0], self._image_shape[1], self._num_img_channels)
        image = image[:, :, 0:3]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if not self._multi_target:
            mask = cv2.inRange(hsv, np.array(self._hsv_mask[0]), np.array(self._hsv_mask[1]))
        else:
            if self._current_target == self._target1:
                mask = cv2.inRange(hsv, np.array(self._target1_hsv[0]), np.array(self._target1_hsv[1]))
            else:
                mask = cv2.inRange(hsv, np.array(self._target2_hsv[0]), np.array(self._target2_hsv[1]))

        # output = cv2.bitwise_and(image, image, mask=mask)
        # gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY) # original rgb2gray
        # _, blackAndWhiteImage = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.fillPoly(blackAndWhiteImage, pts=contours, color=(255, 255, 255))
        # target_size = np.sum(blackAndWhiteImage/255.) / blackAndWhiteImage.size
        
        kernel = np.ones((3, 3), 'uint8')
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)

        target_size = np.sum(mask/255.) / mask.size
        
        #print('target size:', target_size)
        if target_size >= self._min_target_size:
            done = 1

        if self._dense_reward:
            reward = (2.0/(1+np.exp(-target_size*10.0))) - 1.0 
            reward -= 1

        if self._ob_type == OB_TYPE_1 or self._ob_type == OB_TYPE_3:
            self._image_buffer.append(np.concatenate((image, np.expand_dims(mask, axis=-1)), axis=2))
        else:
            self._image_buffer.append(image)

        return reward, done

    def _write_opcode(self, opcode_name, *args):
        """Helper method to force write a command not part of the action dimension.
        Args:
            opcode_name:    the name of the opcode
            *args:          any arguments require for the operation
        """
        # write the command directly to actuator_buffer to avoid the limitation that the opcode
        # is not part of the action dimension
        self._actuator_comms['Create2'].actuator_buffer.write(
            np.concatenate(([create2_config.OPCODE_NAME_TO_CODE[opcode_name]], np.array(args).astype('i'))))

    def _wait_until_charged(self):
        """Waits until Create 2 is sufficiently charged."""
        sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()
        logging.info("Need to charge .. {}.".format(sensor_window[-1]['battery charge']))
        while sensor_window[-1][0]['battery charge'] < self._max_battery:
            # move it out of the dock to avoid the weird non-responsive sleep mode (the non-responsive sleep
            # mode can happen on any mode while on the dock, but only detectable when in PASSIVE mode)
            self._write_opcode('safe')
            time.sleep(0.1)
            self._write_opcode('seek_dock')
            time.sleep(0.1)
            logging.info("Create2 charging with current charge at {}.".format(sensor_window[-1]['battery charge']))
            time.sleep(10)
            print('current charge:', sensor_window[-1][0]['battery charge'])
            sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()

    # ======== rllab compatible gym codes =========

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def image_space(self):
        return self._image_space

    @property
    def proprioception_space(self):
        return self._observation_space

    def terminate(self):
        super().close()

# def random_policy_done_2_done_length():
#     seed = 1
#     np.random.seed(seed)
#     total_dones = 50
#     task = "create2 visual reacher"
#     episode_length_time = 30.0
#     dt = 0.045
#     env = Create2VisualReacherEnv(episode_length_time=episode_length_time, dt=dt, image_shape=(9, 120, 160), camera_id=0, min_target_size=0.2)
#     env.start()

#     # Experiment
#     timeout = int(episode_length_time/dt)
#     done_2_done_lens = []
#     steps = 0
#     while len(done_2_done_lens) < total_dones:
#         env.reset()
#         epi_steps = 0
#         done = 0
#         done_2_done_steps = 0
#         resets = 0
#         while not done:
#             action = env.action_space.sample()

#             # step in the environment
#             _, _, done, _ = env.step(action)

#             # Log
#             steps += 1
#             epi_steps += 1
#             done_2_done_steps += 1

#             # Termination
#             if epi_steps == timeout:
#                 resets += 1
#                 env.reset()
#                 epi_steps = 0

#         done_2_done_lens.append(done_2_done_steps)
 
#         print('Episode: {}, done_2_done steps: {},resets: {}, total steps: {}'.format(len(done_2_done_lens), done_2_done_steps, resets, steps))


#     with open(task + "_random_stat.txt") as out_file:
#         for length in done_2_done_lens:
#             out_file.write(str(length)+'\n')

#         out_file.write(f"\nMean: {mean(done_2_done_lens)}")
        
# if __name__ == '__main__':
#     # env = Create2VisualReacherEnv(episode_length_time=60, dt=0.045, image_shape=(9, 120, 160), camera_id=0, min_target_size=0.1)
#     # env.start()

#     # env.reset()
#     # for i in range(10000):
#     #     a = env.action_space.sample()
#     #     (image, _), _, done, _ = env.step([0,0])
        
#     #     image = np.transpose(image, [1, 2, 0])
#     #     cv2.imshow('', image[:,:,0:3])
#     #     cv2.waitKey
#     #     print(i+1, done)
#     random_policy_done_2_done_length()


# if __name__ == '__main__':
#     env = Create2VisualReacherEnv(episode_length_time=15, dt=0.015, image_shape=(9, 160, 90), camera_id=0, min_target_size=0.1)
#     env.start()

#     env.reset()
#     for i in range(10):
#         print('here 1')
#         a = env.action_space.sample()
#         print('here 2')
#         (image, obs), _, done, _ = env.step(a)

#         print(f'i: {i}, obs: {obs}, im_shape: {image.shape}, action: {a}')
        
#         cv2.imwrite(f'img_{i}.jpg', image[:,:,0:3])

#     env.close()