'''
TODO rebuild my lic codes from numpy version to taichi version
1. parallel
    use taichi
2. 2d LIC first
# ! give up the class and function, take Process-oriented program 

'''


import numpy as np
from scipy.interpolate import interpn
import taichi as ti 
import taichi_glsl as ts
ti.init(arch = ti.gpu)

'''
with h5py.open() as f:
    B_x = ...
    B_y = ...
'''

field_shape = (100, 100)
 






if __name__ == "__main__":
    shape = (10,10,10)
    np.random.seed(1)
    # ux = np.random.rand(5, 6, 7)
    ux = np.ones(shape)
    np.random.seed(3)
    uy = np.random.rand(shape[0], shape[1], shape[2])
    np.random.seed(10)
    uz = np.random.rand(shape[0], shape[1], shape[2])
    test_field = vectorfield(shape, ux, ux, ux)
    P = ti.Vector([1,1.5,2])
    
    # print(test_field.field_point('x', P))
    @ti.kernel
    def main() -> ti.f64:
        # x =  ts.sampling.sample(field, P)
        x = test_field.field_point_x(P)

        return x
    print(main())


