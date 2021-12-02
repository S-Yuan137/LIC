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
import taichi_glsl as ts
ti.init(arch = ti.gpu)

def normalize(ux, uy, uz):
    mag = np.sqrt(ux**2 + uy**2 + uz**2)
    return ux/mag, uy/mag, uz/mag

@ti.func
def trilerp(field: ti.template(), P):
    '''
    Tilinear sampling an 3D field with a real index.
    :parameter field: (3D Tensor)
        Specify the field to sample.
    :parameter P: (3D Vector of float)
        Specify the index in field.
    :note:
        If one of the element to be accessed is out of `field.shape`, then
        `Tilerp` will automatically do a clamp for you, see :func:`sample`.
        Syntax ref : https://en.wikipedia.org/wiki/Trilinear_interpolation.
    :return:
        The return value is calcuated as::
            I = int(P)
            w0 = ts.fract(P)
            w1 = 1.0 - w0
            c00 = ts.sample(field, I + ts.D.yyy) * w1.x + ts.sample(field, I + ts.D.xyy) * w0.x
            c01 = ts.sample(field, I + ts.D.yyx) * w1.x + ts.sample(field, I + ts.D.xyx) * w0.x
            c10 = ts.sample(field, I + ts.D.yxy) * w1.x + ts.sample(field, I + ts.D.xxy) * w0.x
            c11 = ts.sample(field, I + ts.D.yxx) * w1.x + ts.sample(field, I + ts.D.xxx) * w0.x
            c0 = c00 * w1.y + c10 * w0.y
            c1 = c01 * w1.y + c11 * w0.y
            return c0 * w1.z + c1 * w0.z
        .. where D = vec(1, 0, -1)
    '''
    I = int(P)
    w0 = ts.fract(P)
    w1 = 1.0 - w0

    c00 = ts.sample(field, I + ts.D.yyy) * w1.x + ts.sample(
        field, I + ts.D.xyy) * w0.x
    c01 = ts.sample(field, I + ts.D.yyx) * w1.x + ts.sample(
        field, I + ts.D.xyx) * w0.x
    c10 = ts.sample(field, I + ts.D.yxy) * w1.x + ts.sample(
        field, I + ts.D.xxy) * w0.x
    c11 = ts.sample(field, I + ts.D.yxx) * w1.x + ts.sample(
        field, I + ts.D.xxx) * w0.x

    c0 = c00 * w1.y + c10 * w0.y
    c1 = c01 * w1.y + c11 * w0.y

    return c0 * w1.z + c1 * w0.z

@ti.data_oriented
class vectorfield: 
    # in light of the shortcoming of taichi return, one must normalize the field before initializing class
    def __init__(self, size, Vx: ti.ext_arr(), Vy: ti.ext_arr(), Vz: ti.ext_arr()):
        # the vectorfield is (Vx, Vy, Vz), each component has the same 3D size
        self.size = size
        self.Vx   = Vx
        self.Vy   = Vy
        self.Vz   = Vz
        self.licTexture = ti.field(dtype = ti.f64, shape = self.size)
    
    @staticmethod
    @ti.pyfunc
    def distance(Coord1, Coord2):
        return np.sqrt(np.sum((np.array(Coord1) - np.array(Coord2))**2))

    # @staticmethod
    # @ti.pyfunc
    # def trilerp(field: ti.template(), coord):
    #     '''
    #     Tilinear sampling an 3D field with a real index.
    #     :parameter field: (3D Tensor)
    #         Specify the field to sample.
    #     :parameter P: (3D Vector of float)
    #         Specify the index in field.
    #     :note:
    #         If one of the element to be accessed is out of `field.shape`, then
    #         `Tilerp` will automatically do a clamp for you, see :func:`sample`.
    #         Syntax ref : https://en.wikipedia.org/wiki/Trilinear_interpolation.
    #     :return:
    #         The corresponding value (scalar)
    #     '''
    #     I = int(coord)
    #     w0 = ts.fract(coord)
    #     w1 = 1.0 - w0
    #     c00 = ts.sample(field, I + ts.D.yyy) * w1.x + ts.sample(field, I + ts.D.xyy) * w0.x
    #     c01 = ts.sample(field, I + ts.D.yyx) * w1.x + ts.sample(field, I + ts.D.xyx) * w0.x
    #     c10 = ts.sample(field, I + ts.D.yxy) * w1.x + ts.sample(field, I + ts.D.xxy) * w0.x
    #     c11 = ts.sample(field, I + ts.D.yxx) * w1.x + ts.sample(field, I + ts.D.xxx) * w0.x
    #     c0 = c00 * w1.y + c10 * w0.y
    #     c1 = c01 * w1.y + c11 * w0.y
    #     return c0 * w1.z + c1 * w0.z

    @ti.pyfunc
    def in_field(self, coord) -> bool:
        non_neg = coord[0] >=0 and coord[1]>=0 and coord[2]>=0
        in_size = coord[0] <= self.size[0]-1 and coord[1] <= self.size[1]-1 and coord[2] <= self.size[2]-1
        return non_neg and in_size

    # @ti.func 
    # def field_point(self, component, coord) -> ti.f64:
    #     if component == 'x':
    #         x = trilerp(self.Vx, coord)
    #         return x
    #     elif component == 'y':
    #         return trilerp(self.Vy, coord)
    #     elif component == 'z':
    #         return trilerp(self.Vz, coord)
    #     else:
    #         pass

    @ti.func
    def field_point_x(self, coord):
        field = self.Vx
        return trilerp(field, coord)






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

        print(x)
    main()


