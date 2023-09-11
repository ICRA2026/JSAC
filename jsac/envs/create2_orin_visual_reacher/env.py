# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
import gym
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
import numpy as np
import jsac.envs.create2_orin_visual_reacher.senseact_create_env.create2_config as create2_config
from jsac.envs.create2_orin_visual_reacher.senseact_create_env import utils

from multiprocessing import Array, Value

from jsac.envs.create2_orin_visual_reacher.senseact_create_env.rtrl_base_env import RTRLBaseEnv
from jsac.envs.create2_orin_visual_reacher.senseact_create_env.create2_communicator import Create2Communicator
from jsac.envs.create2_orin_visual_reacher.senseact_create_env.create2_observation import Create2ObservationFactory

from jsac.envs.create2_orin_visual_reacher.fast_cam import FastCamera
import cv2
from statistics import mean

class Create2VisualReacherEnv(RTRLBaseEnv, gym.Env):
    """Create2 environment for training it drive forward.
    By default this environment observes the infrared character (binary state for detecting the dock),
    bump sensor, and last action.
    The reward is the forward reward.  Episode ends early when the bump sensor is triggered.
    TODO:
        * move all common methods between this class and docking_env to a base class
    """

    def __init__(self, episode_length_time=15, port='/dev/ttyUSB0', obs_history=1, dt=0.015, image_shape=(0, 0, 0),
                 camera_id=0, min_target_size=0.1, pause_before_reset=0, pause_after_reset=0, **kwargs):
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
        self._hsv_mask = ((30, 60, 80), (90, 180, 255))
        self._min_target_size = min_target_size
        self._min_battery = 800
        self._max_battery = 1800

        # get the opcode for our main action (only 1 action)
        self._main_op = 'drive_direct'
        self._extra_ops = ['safe', 'seek_dock', 'drive']
        main_opcode = create2_config.OPCODE_NAME_TO_CODE[self._main_op]
        extra_opcodes = [create2_config.OPCODE_NAME_TO_CODE[op] for op in self._extra_ops]

        # store the previous action/reward to be shared across processes
        self._prev_action_ = np.frombuffer(Array('i', 2).get_obj(), dtype='i')

        # create factory with common arguments for making an observation dimension
        observation_factory = Create2ObservationFactory(main_op=self._main_op,
                                                        dt=dt,
                                                        obs_history=self._obs_history,
                                                        internal_timing=self._internal_timing,
                                                        prev_action=self._prev_action_)

        # the definition of the observed state and the associated custom modification (if any)
        # before passing to the learning algorithm
        self._observation_def = [
            observation_factory.make_dim('light bump left signal'),
            observation_factory.make_dim('light bump front left signal'),
            observation_factory.make_dim('light bump center left signal'),
            observation_factory.make_dim('light bump center right signal'),
            observation_factory.make_dim('light bump front right signal'),
            observation_factory.make_dim('light bump right signal'),
            observation_factory.make_dim('previous action')
        ]

        # extra packets we need for proper reset and charging
        self._extra_sensor_packets = ['bumps and wheel drops', 'battery charge',
                                      'oi mode', 'distance','charging sources available',
                                      'cliff left', 'cliff front left', 'cliff front right', 'cliff right']
        
        packet_names = ['light bump left signal', 'light bump front left signal', 
        'light bump center left signal', 'light bump center right signal', 
        'light bump front right signal', 'light bump right signal', ] +  self._extra_sensor_packets

        main_sensor_packet_ids = [d.packet_id for d in self._observation_def if d.packet_id is not None]
        extra_sensor_packet_ids = [create2_config.PACKET_NAME_TO_ID[nm] for nm in self._extra_sensor_packets]

        # TODO: move this out to some base class?
        from gym.spaces import Box as GymBox  # use this for baselines algos
        Box = GymBox

        # go thru the main opcode (just direct_drive in this case) and add the range of each param
        # XXX should the action space include the opcode? what about op that doesn't have parameters?
        self._action_space = Box(
            low=np.array([r[0] for r in create2_config.OPCODE_INFO[main_opcode]['params'].values()]),
            high=np.array([r[1] for r in create2_config.OPCODE_INFO[main_opcode]['params'].values()])
        )

        # loop thru the observation dimension and get the lows and highs
        self._observation_space = Box(
            low=np.concatenate([d.lows for d in self._observation_def]),
            high=np.concatenate([d.highs for d in self._observation_def])
        )

        self._comm_name = 'Create2'
        
        self._comms_packet_names = {}
        self._comms_packet_names[self._comm_name] = packet_names

        communicator_setups = {}
        buffer_len = int(dt / self._internal_timing + 1)
        communicator_setups[self._comm_name] = {'Communicator': Create2Communicator,
                                                 # have to read in this number of packets everytime to support
                                                 # all operations
                                                 'num_sensor_packets': buffer_len,
                                                 'kwargs': {'sensor_packet_ids': main_sensor_packet_ids +
                                                                                 extra_sensor_packet_ids,
                                                            'opcodes': [main_opcode] + extra_opcodes,
                                                            'port': port,
                                                           }
                                            }
        
        self._latest_sensor_dict = None

        self._image_shape = image_shape
        self._image_space = Box(low=0, high=255, shape=self._image_shape)

        if self._image_shape != (0, 0, 0):
            height, width, _ = self._image_shape
            self._cam = FastCamera(res=(width, height), device_id=camera_id, dt=dt)
        
        super().__init__(communicator_setups=communicator_setups, dt=dt,
                        **kwargs)

    def _read_sensation(self):
        # overwrite this to support image

        sensor_dict = self._get_sensor_dict()
        r_r, r_d = self._calc_roomba_obs_reward(sensor_dict)
        roomba_obs = []
        for d in self._observation_def:
            res = d.normalized_handler(sensor_dict)
            roomba_obs.extend(res)

        image = None
        im_r = 0
        im_d = 0
        if self._image_shape != (0, 0, 0):
            image = self._cam.get_img()
            self._image_history[:, :, 3:] = self._image_history[:, :, :-3]
            self._image_history[:, :, 0:3] = image

            im_r, im_d = self._calc_image_reward(image)

            done = self._check_done(r_d or im_d)
            return (self._image_history, roomba_obs), r_r+im_r-1,  done
        
        return roomba_obs, r_r, r_d


    def _compute_actuation_(self, action):
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
        action = action.astype(int)
        self._actuation_packet_['Create2'] = np.concatenate(
            ([create2_config.OPCODE_NAME_TO_CODE[self._main_op]], action))
        np.copyto(self._prev_action_, action)


    def _sensor_to_sensation_(self):
        pass

    def _reset_(self):
        """The required _reset_ interface.
        This method does the handling of charging the Create2, repositioning, and set to the correct mode.
        """
        logging.info("Resetting...")
        self._write_opcode('drive', 0, 0)
        time.sleep(0.1)
        # N.B: pause_before_reset should be greater than zero only for demo purposes
        time.sleep(self.pause_before_reset)

        self._episode_step_.value = -1
        np.copyto(self._prev_action_, np.array([0, 0]))
        for d in self._observation_def:
            d.reset()

        # wait for camera to startup properly
        _ = self._cam.get_img()

        # wait for create2 to startup properly if just started (ie. wait to actually start receiving observation)
        sensor_dict = self._get_sensor_dict()

        self._image_history = np.zeros(shape=self._image_shape, dtype=np.uint8)

        print('current charge:', sensor_dict['battery charge'])

        if sensor_dict['battery charge'] <= self._min_battery:
            print("Waiting for Create2 to be docked.")
            if sensor_dict['charging sources available'] <= 0:
                self._write_opcode('drive_direct', 0, 0)
                time.sleep(0.75)
                self._write_opcode('seek_dock')
                time.sleep(10)
            self._wait_until_charged()
            sensor_dict = self._get_sensor_dict()

        # Always switch to SAFE mode to run an episode, so that Create2 will switch to PASSIVE on the
        # charger.  If the create2 is in any other mode on the charger, we will not be able to detect
        # the non-responsive sleep mode that happens at the 60 seconds mark.
        logging.info("Setting Create2 into safe mode.")
        self._write_opcode('safe')
        time.sleep(0.1)

        # after charging/docked, try to drive away from the dock if still on it
        if sensor_dict['charging sources available'] > 0:
            logging.info("Undocking the Create2.")
            self._write_opcode('drive_direct', -250, -250)
            time.sleep(0.75)
            self._write_opcode('drive_direct', 0, 0)
            time.sleep(0.1)

        # rotate and drive backward 
        logging.info("Moving Create2 into position.")
        target_values = [-300, -300]
        move_time_1 = np.random.uniform(low=1, high=1.5)
        move_time_2 = np.random.uniform(low=0.3, high=0.6)
        rotate_time_1 = np.random.uniform(low=0.25, high=0.75)
        rotate_time_2 = np.random.uniform(low=0.5, high=1)
        direction = np.random.choice((1, -1))
        
        # rotate
        self._write_opcode('drive_direct', *(300*direction, -300*direction))
        time.sleep(rotate_time_1)
        self._write_opcode('drive', 0, 0)
        time.sleep(0.1)

        # back
        self._write_opcode('drive_direct', *target_values)
        time.sleep(move_time_1)
        self._write_opcode('drive', 0, 0)
        time.sleep(0.1)

        # rotate
        self._write_opcode('drive_direct', *(300*direction, -300*direction))
        time.sleep(rotate_time_2)
        self._write_opcode('drive', 0, 0)
        time.sleep(0.1)
        
        # back
        self._write_opcode('drive_direct', *[300, 300])
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
        sensor_dict = self._get_sensor_dict()
        while sensor_dict['oi mode'] != 2:
            logging.warning("Create2 not in SAFE mode, reattempting... (might require human intervention).")
            self._write_opcode('full')
            time.sleep(0.2)
            self._write_opcode('drive_direct', -50, -50)
            time.sleep(0.5)
            self._write_opcode('drive', 0, 0)
            time.sleep(0.1)
            self._write_opcode('safe')
            time.sleep(0.2)

            sensor_dict = self._get_sensor_dict()

        # N.B: pause_after_reset should be greater than zero only for demo purposes
        time.sleep(self.pause_after_reset)

        # don't want the state during reset pollute the first sensation
        time.sleep(2 * self._internal_timing)

        print("Reset completed.")

    def _get_sensor_dict(self):
        while self._sensor_comms[self._comm_name].sensor_buffer.empty() and self._latest_sensor_dict == None:
            time.sleep(0.0001)

        if not self._sensor_comms[self._comm_name].sensor_buffer.empty(): 
            sensor_window = self._sensor_comms[self._comm_name].sensor_buffer.get()
            while not self._sensor_comms[self._comm_name].sensor_buffer.empty():
                sensor_window = self._sensor_comms[self._comm_name].sensor_buffer.get()
            
            sensor_dict = {}
            for idx, packet_name in enumerate(self._comms_packet_names[self._comm_name]):
                sensor_dict[packet_name] = sensor_window[-1][idx]
            
            self._latest_sensor_dict = sensor_dict
            return self._latest_sensor_dict
        
        return self._latest_sensor_dict
        

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
        
    def _calc_roomba_obs_reward(self, obs):
        """Helper to calculate reward.
        Args:
            sensor_window: the sensor_window from _compute_sensation_
        Returns:
            A tuple of (reward, done)
        """
        
        bw = 0
        for p in range(int(self._dt / self._internal_timing)):
            bw |= obs['bumps and wheel drops']
        
        cl = 0
        for p in range(int(self._dt / self._internal_timing)):
            cl += obs['cliff left']
            cl += obs['cliff front left']
            cl += obs['cliff front right']
            cl += obs['cliff right']
        
        reward = 0

        charging_sources_available = obs['charging sources available']

        oi_mode = obs['oi mode']
        if oi_mode == 1 and charging_sources_available == 0 and cl == 0:
            self._write_opcode('safe')

        # If wheel dropped, it's done and result in a big penalty.
        done = 0
        if (bw >> 2) > 0:
            done = 1
            reward = -self._episode_length_step

        return reward, done

    def _calc_image_reward(self, image):
        reward = 0.0
        done = 0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self._hsv_mask[0]), np.array(self._hsv_mask[1]))
        output = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY) # original rgb2gray
        _, blackAndWhiteImage = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(blackAndWhiteImage, pts=contours, color=(255, 255, 255))
        target_size = np.sum(blackAndWhiteImage/255.) / blackAndWhiteImage.size
        #print('target size:', target_size)
        if target_size >= self._min_target_size:
            done = 1

        return reward, done

    def _write_opcode(self, opcode_name, *args):
        """Helper method to force write a command not part of the action dimension.
        Args:
            opcode_name:    the name of the opcode
            *args:          any arguments require for the operation
        """
        # write the command directly to actuator_buffer to avoid the limitation that the opcode
        # is not part of the action dimension
        self._actuator_comms[self._comm_name].actuator_buffer.put(
            np.concatenate(([create2_config.OPCODE_NAME_TO_CODE[opcode_name]], np.array(args).astype('i'))))

    def _wait_until_charged(self):
        """Waits until Create 2 is sufficiently charged."""
        sensor_dict = self._get_sensor_dict()
        logging.info("Need to charge .. {}.".format(sensor_dict['battery charge']))
        while sensor_dict['battery charge'] < self._max_battery:
            # move it out of the dock to avoid the weird non-responsive sleep mode (the non-responsive sleep
            # mode can happen on any mode while on the dock, but only detectable when in PASSIVE mode)
            self._write_opcode('safe')
            time.sleep(0.1)
            self._write_opcode('seek_dock')
            time.sleep(0.1)
            logging.info("Create2 charging with current charge at {}.".format(sensor_dict['battery charge']))
            time.sleep(10)
            print('current charge:', sensor_dict['battery charge'])
            sensor_dict = self._get_sensor_dict()

    # def _get_indices_from_packet_names(self, names):
    #     indices={}
    #     for idx, name in enumerate(names):
    #         indices[name] = idx
    #     return indices

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

    def close(self):
        self._cam.close()
        super().close()
        
if __name__ == '__main__':
    env = Create2VisualReacherEnv(episode_length_time=15, dt=0.015, image_shape=(90, 160, 9), camera_id=0, min_target_size=0.1)
    env.start()

    env.reset()
    for i in range(10):
        a = env.action_space.sample()
        (image, obs), _, done, _ = env.step(a)

        print(f'i: {i}, obs: {obs}, im_shape: {image.shape}, action: {a}')
        
        cv2.imwrite(f'img_{i}.jpg', image[:,:,0:3])

    env.close()

    # sudo chmod a+rw /dev/ttyUSB0