'''
TODO rebuild my lic codes from numpy version to taichi version
1. parallel
    use taichi
2. dimension independent (automatically support 2d and 3d)
    use taichi metaprogramming

# ? it seems to be not necessary to create a class
# ! still tey use class as an exercise

'''


import numpy as np
from scipy.interpolate import interpn
import taichi as ti 
ti.init(arch = ti.gpu)












