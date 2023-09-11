# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
import logging
import numpy as np

from jsac.envs.create2_orin_visual_reacher.senseact_create_env.create2_api import Create2
from jsac.envs.create2_orin_visual_reacher.senseact_create_env.communicator import Communicator

class Create2Communicator(Communicator):
    """The class implements communicator for Create2 device."""
    def __init__(self, sensor_packet_ids, opcodes, port='/dev/ttyUSB0', 
                 baudrate=115200):
        """Inits the communicator object with device-specific parameters.

        Args:actuator_buffer
            sensor_packet_ids: a list of integer packet ids we are interested in
            opcodes: a list of integer opcodes of the action we will be taking
            port: a string specifying the port of the serial connection
            baudrate: an integer specifying the baudrate to connect at
        """
        self._create2 = Create2()
        self._sensor_packet_ids = sensor_packet_ids
        self._port = port
        self._baudrate = baudrate

        # gather info for total sensor packets size
        sensor_packet_dtypes = []
        for packet_id in sensor_packet_ids:
            sensor_packet_dtypes.extend(self._create2.get_packet_dtype(packet_id).descr)
        final_sensor_dtype = np.dtype(sensor_packet_dtypes)

        # gather info for max actuator command parameters size
        # TODO support the song command with variable parameter list when implemented
        max_params = 0
        for opcode in opcodes:
            max_params = max(max_params, len(self._create2.get_op_params(opcode)))

        super(Create2Communicator, self).__init__(use_sensor=True, use_actuator=True)

    def run(self):
        """Method called by Python when the communicator process is started."""
        # connect in subprocess since IO can't be shared across processes on all OS
        self._connect()

        # entering main communicator process loop
        super(Create2Communicator, self).run()

        self._disconnect()

    def _sensor_handler(self):
        """Implement the main sensor handler for the communicator.

        Read as fast as possible, let the checksum check catch any corrupted data.  Or
        we could limit to 15 ms, which is how often Create2 update its state.
        """
        # If the Create2 is on the charger in PASSIVE mode, it will enter into a deep
        # sleep after 60 seconds in which the stream packet stops.  The normal wake
        # command only wake it for around 300ms.  Only a full reconnect seems to be
        # able to wake it for another 60 seconds.  If the Create2 is at the dock in
        # non-PASSIVE mode, it will silently enter the deep sleep mode but with stream
        # packets continuing.  Any opcode send to the Create2 in this state will be
        # non-responsive.  Therefore it is best to always put the Create2 into PASSIVE
        # when on the dock so we can detect the deep sleep.
        #
        # Alternatively, drive the Create2 out of the dock before the 60 seconds timer
        # to prevent the deep non-responsive sleep from happening.
        try:
            data = self._create2.get_stream_packets()
        except IOError:
            # Keep trying to reconnect until succeed or when we are stopping the communicator.
            # This can takes as many as a dozen tries or as little as one try.  As of now
            # there's no clear explanation for this behavior.
            success = False
            while not success and self._sensor_running:
                logging.info("Attempting to reconnect Create2 communicator.")
                try:
                    self._reconnect()
                    success = True
                except Exception:
                    pass
            return

        self.sensor_buffer.put(data)

    def _actuator_handler(self):
        """Implements the main actuator handler for the communicator."""
        if not self.actuator_buffer.empty():
            recent_actuation = self.actuator_buffer.get()
            opcode = recent_actuation[0]
            args = recent_actuation[1:]

            # just let the action evaporate into the ether if we are reconnecting; we
            # don't want to hold onto actions that are too old
            try:
                self._create2.exec_op(opcode, *args)
            except Exception:
                pass

        time.sleep(1e-4)

    def _connect(self):
        """Helper method to connect to the Create2.

        Includes all the operations needed to start the Create2 up properly for RL.
        """
        self._create2.connect(port=self._port, baudrate=self._baudrate)

        # Sometimes the Create2 needs a reset, sometimes not, and sometimes
        # Create2 does not response to a reset command.  Reset here just
        # in case.  It should play several beeps if successfully reset.
        self._create2.reset()

        # Start the Create2 open interface and put it in PASSIVE mode.
        self._create2.start()
        self._create2.verify_mode()

        # begin the sensor packets streaming
        self._create2.stream(self._sensor_packet_ids)

    def _disconnect(self):
        """Helper method to disconnect from the Create2.

        Includes all the operations to fully close the Create2.
        """
        # pause the stream and seek dock
        self._create2.pause_resume_stream(0)
        self._create2.seek_dock()

        time.sleep(1.0)

        # Put the Create2 into OFF mode (but sometimes it ended up in PASSIVE).
        self._create2.stop()
        self._create2.disconnect()

    def _reconnect(self):
        """Helper method for reconnect to the Create2.

        Called when there is trouble with the connection (such as sleeping at the dock).
        Note that if the Create2 was not in PASSIVE mode before, its mode will be
        switched back to PASSIVE.
        """
        try:
            self._disconnect()
        except Exception:
            # don't care if disconnect failed
            pass
        time.sleep(1.0)
        self._connect()
