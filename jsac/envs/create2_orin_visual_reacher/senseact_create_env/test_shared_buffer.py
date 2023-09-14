from multiprocessing import Array, Process
import multiprocessing as mp
import random
import time
from sharedbuffer import SharedBuffer

def fnc(vars):
    arr = SharedBuffer(4, 'i', int, True)
    arr.set_state(vars)
    for i in range(10):
        arr.write(i+5)
        time.sleep(0.1)
    
    time.sleep(1)
    print()
    for i in range(10):
        print(arr.read_update())


if __name__ == '__main__':
    mp.set_start_method('spawn')
    arr = SharedBuffer(4, 'i', int, True)
    vars = arr.get_state()
    p = Process(target=fnc, args=(vars,))
    p.start()
    time.sleep(0.1)

    for i in range(10):
        print(arr.read_update())

        time.sleep(0.1)
    
    p.join()