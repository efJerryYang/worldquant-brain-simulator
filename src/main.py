from simulator.simulate import Simulator
from alpha_pool.alpha import *

if __name__ == '__main__':
    # avg: 15s for multiprocessing
    # avg: 36s for single process
    s = Simulator()
    s.simulate_with_multiprocessing(eg_alpha)
    # s.simulate_with_multiprocessing(eg_alpha2)
    # s.simulate_with_multiprocessing(eg_alpha3)
    # s.simulate(eg_alpha3)