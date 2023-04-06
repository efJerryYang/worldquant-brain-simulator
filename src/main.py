from simulator.simulate import Simulator
from alpha_pool.alpha import *

s = Simulator()

# s.simulate_with_multiprocessing(eg_alpha)
s.simulate(eg_alpha)