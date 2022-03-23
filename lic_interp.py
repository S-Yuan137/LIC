import numpy as np
from scipy.interpolate import interpn
import h5py
from numba import njit, jit
import time
import matplotlib.pyplot as plt

from interpolation.splines import eval_linear, UCGrid


@njit
def interp2d(field, Indexcoord):
    grid = UCGrid((0, field.shape[0] - 1, field.shape[0]), (0, field.shape[1] - 1, field.shape[1]))
    return eval_linear(grid, field, Indexcoord)


def lic_2(vector_field_x, vector_field_y, dt=0.5, len_pix=5, t=0, noise=None):
    def interp2d(field, Indexcoord):
        grid = UCGrid((0, field.shape[0] - 1, field.shape[0]), (0, field.shape[1] - 1, field.shape[1]))
        return eval_linear(grid, field, Indexcoord)

    vector_field_x = np.asarray(vector_field_x)
    vector_field_y = np.asarray(vector_field_y)
    assert vector_field_x.shape == vector_field_y.shape
    # ========= normalization ===============
    vector_field_x = vector_field_x / np.sqrt(vector_field_x**2 + vector_field_y**2)
    vector_field_y = vector_field_y / np.sqrt(vector_field_x**2 + vector_field_y**2)
    m, n = vector_field_x.shape
    if noise is None:
        noise = np.random.rand(*(vector_field_x.shape))
    result = np.zeros(vector_field_x.shape)

    def in_field(x, y):
        non_neg = x >= 0 and y >= 0

        in_size = (x <= m - 1) and y <= n - 1

        return non_neg and in_size

    print(in_field(0, 0))

    for i in np.arange(0, m - 1, 1):
        for j in np.arange(0, n - 1, 1):

            # upward streamline
            x = i
            y = j
            arc_len = 0

            stream_up = 0
            c = 0
            forward_total = 0
            while arc_len < len_pix and in_field(x, y) and c < 500:

                vx = interp2d(vector_field_x, (x, y))
                vy = interp2d(vector_field_y, (x, y))
                T_noise = interp2d(noise, (x, y))
                weight = (np.cos(t + 0.46 * arc_len)) ** 2
                stream_up = stream_up + T_noise * weight
                x = x + vx * dt
                y = y + vy * dt
                arc_len = arc_len + np.sqrt((vx * dt) ** 2 + (vy * dt) ** 2)
                c = c + 1
                forward_total += weight

            # downward streamline
            x = i
            y = j
            arc_len = 0
            stream_down = 0
            c = 0
            backward_total = 0
            while arc_len < len_pix and in_field(x, y) and c < 500:
                # notice Not compute the origin point twice!
                vx = interp2d(vector_field_x, (x, y))
                vy = interp2d(vector_field_y, (x, y))
                x = x - vx * dt
                y = y - vy * dt
                T_noise = interp2d(noise, (x, y))
                weight = np.cos(t - 0.46 * arc_len) ** 2
                stream_down = stream_down + T_noise * np.cos(t - 0.46 * arc_len) ** 2
                arc_len = arc_len + np.sqrt((vx * dt) ** 2 + (vy * dt) ** 2)
                c = c + 1
                backward_total += weight
            result[i, j] = stream_up + stream_down / (forward_total + backward_total)
    return result


def show_grey(tex):
    plt.figure()
    tex = tex.T
    plt.imshow(tex, origin="lower", cmap="Greys")


def streamlines(Vx, Vy):
    Vx, Vy = Vx.T, Vy.T
    x = np.linspace(0, Vx.shape[0] - 1, Vx.shape[0])
    y = np.linspace(0, Vx.shape[1] - 1, Vx.shape[1])
    x, y = np.meshgrid(y, x)
    print(x.shape, y.shape, Vx.shape)

    plt.streamplot(x, y, Vx, Vy, density=1)


if __name__ == "__main__":
    with h5py.File(r"D:\CUHK\Data_from_zcao\struct01\struct01_snap52.h5", "r") as f:
        B_x = f["i_mag_field"][:, :, 50]
        B_y = f["j_mag_field"][:, :, 50]
        rho = f["gas_density"][:, :, 50]
    start_time = time.time()
    # B_x = bilinear_interpolation(B_x, 5)
    # B_y = bilinear_interpolation(B_y, 5)
    # show_grey(lic_2d(B_x, B_y, t=0, len_pix=10))
    show_grey(lic_2(B_x, B_y, dt=0.1, len_pix=20, t=0))

    streamlines(B_x, B_y)
    print()
    print("--- %.2f seconds ---" % (time.time() - start_time))
    plt.show()
