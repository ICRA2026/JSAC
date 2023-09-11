import cv2
import time
import numpy as np
from multiprocessing import Queue, Process
from threading import Thread, Lock

class FastCamera:
    def __init__(self, res, device_id=0, dt=0):
        self._res = res  # (width, height)
        self._device_id = device_id
        self._dt = dt

        self._img_queue = Queue()
        self._info_queue = Queue()

        self._img_process = Process(target=self._async_img_capture)
        self._img_process.start()

        self._last_img = None
        self._img_ready = False

        self._img_lock = Lock() 
        self._img_idx = 0

        self._rcv_img_thread = Thread(target=self._rcv_img)
        self._rcv_img_thread.start()

    def _rcv_img(self):
        while True:
            data = self._img_queue.get()

            if isinstance(data, str):
                if data == 'exit':
                    return
                else:
                    continue
                
            with self._img_lock:
                self._img_idx += 1
                self._last_img = data
                self._img_ready = True

    def get_img(self):
        # image shape: (height, width, channel)
        while not self._img_ready:
            time.sleep(0.001)
        with self._img_lock:
            self._img_ready = False
            return self._last_img
    
    def close(self):
        self._img_queue.put('exit')
        self._info_queue.put('exit')

        self._img_process.join()
        self._rcv_img_thread.join()

    def _async_img_capture(self):
        cap = cv2.VideoCapture(self._device_id, cv2.CAP_V4L2)

        if not cap.isOpened():
            error_msg = f'Unable to open camera on device id {self._device_id}.'
            raise IOError(error_msg)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 120)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        # Exposure set to manual mode
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 

        # Exposure absolute value; min=0, max=10000, cam_default=312
        cap.set(cv2.CAP_PROP_EXPOSURE, 130)

        # Signal amplification value. min=1, max=40, cam_default=1  
        cap.set(cv2.CAP_PROP_GAIN, 5)

        # Brightness value.  min=-15, max=15, cam_default=0                        
        cap.set(cv2.CAP_PROP_BRIGHTNESS , 2)

        # Contrast value. min=0, max=30, cam_default=9   
        cap.set(cv2.CAP_PROP_CONTRAST , 8)     

        # Fire up the camera
        for _ in range(5):
            ret, frame = cap.read()

        if self._dt > 0:
            self._dt -= 0.0001
            self._step_end_time = time.time() + self._dt

        while True:
            if not self._info_queue.empty():
                msg = self._info_queue.get()
                if msg == 'exit':
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            ret, frame = cap.read()
            frame = cv2.resize(frame, self._res)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            self._img_queue.put(frame)

            if self._dt > 0:
                sleep_time = self._step_end_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self._step_end_time = time.time() + self._dt


# if __name__ == '__main__':
#     fcam = FastCamera(res=(160, 90), dt=0.015)
    
#     times = []  
#     t1 = time.time()
    
#     for i in range(20):
#         img = fcam.get_img()
#         t2 = time.time()
#         times.append((t2 - t1) * 1000)
#         t1 = t2
#         print(img.shape, img.dtype)

#         file_name = f'images/img_{i}.jpg'
#         cv2.imwrite(file_name, img)

#     fcam.close()

#     for i in range(len(times)):
#         print("{:.4f} ".format(times[i]), end='') 
#         if (i+1)%10 == 0:
#             print()
