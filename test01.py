import numpy as np
import taichi as ti
import taichi_glsl as ts
ti.init(arch = ti.gpu)

shape = (5, 5, 6)
field =  ti.field(ti.f64, shape = shape)
np.random.seed(3)
self = np.random.rand(shape[0], shape[1], shape[2])
field.from_numpy(self)
P = ti.Vector([1,1.5,2.005])



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


@ti.kernel
def main() -> ti.f64:
    # x =  ts.sampling.sample(field, P)
    x = trilerp(field, P)

    return x
# val = main()
# print(val)
print(main())