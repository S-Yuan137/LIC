import numpy as np
import h5py
from numba import njit
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import os.path as path
import cv2
import shutil


def get_noise(field_component):
    return np.random.rand(*(field_component.shape))


def bilinear_interpolation(data_in, resample_factor):
    from scipy import interpolate

    x_grid = np.linspace(0, data_in.shape[1] - 1, data_in.shape[1])
    y_grid = np.linspace(0, data_in.shape[0] - 1, data_in.shape[0])
    f = interpolate.interp2d(x_grid, y_grid, data_in, kind="linear")
    out = f(
        np.linspace(0, data_in.shape[1], data_in.shape[1] * resample_factor),
        np.linspace(0, data_in.shape[0], data_in.shape[0] * resample_factor),
    )
    return out


@njit
def lic_2d(vector_field_x, vector_field_y, t=0, len_pix=5, noise=None):
    # here the meaning of noise as an arguement is that one can customize noise
    # dispite of the compatibility of numba
    vector_field_x = np.asarray(vector_field_x)
    vector_field_y = np.asarray(vector_field_y)

    # ========= normalization ===============
    vector_field_x = vector_field_x / np.sqrt(vector_field_x**2 + vector_field_y**2)
    vector_field_y = vector_field_y / np.sqrt(vector_field_x**2 + vector_field_y**2)

    assert vector_field_x.shape == vector_field_y.shape
    m, n = vector_field_x.shape
    if noise is None:
        noise = np.random.rand(*(vector_field_x.shape))

    result = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            y = i
            x = j
            forward_sum = 0
            forward_total = 0
            # Advect forwards
            for k in range(len_pix):
                dx = vector_field_y[int(y), int(x)]
                dy = vector_field_x[int(y), int(x)]
                dt_x = dt_y = 0
                if dy > 0:
                    dt_y = ((np.floor(y) + 1) - y) / dy
                elif dy < 0:
                    dt_y = (y - (np.ceil(y) - 1)) / -dy
                if dx > 0:
                    dt_x = ((np.floor(x) + 1) - x) / dx
                elif dx < 0:
                    dt_x = (x - (np.ceil(x) - 1)) / -dx
                if dx == 0 and dy == 0:
                    dt = 0
                else:
                    dt = min(dt_x, dt_y)
                x = min(max(x + dx * dt, 0), n - 1)
                y = min(max(y + dy * dt, 0), m - 1)
                weight = pow(np.cos(t + 0.46 * k), 2)
                forward_sum += noise[int(y), int(x)] * weight
                forward_total += weight
            y = i
            x = j
            backward_sum = 0
            backward_total = 0
            # Advect backwards
            for k in range(1, len_pix):
                dx = vector_field_y[int(y), int(x)]
                dy = vector_field_x[int(y), int(x)]
                dy *= -1
                dx *= -1
                dt_x = dt_y = 0
                if dy > 0:
                    dt_y = ((np.floor(y) + 1) - y) / dy
                elif dy < 0:
                    dt_y = (y - (np.ceil(y) - 1)) / -dy
                if dx > 0:
                    dt_x = ((np.floor(x) + 1) - x) / dx
                elif dx < 0:
                    dt_x = (x - (np.ceil(x) - 1)) / -dx
                if dx == 0 and dy == 0:
                    dt = 0
                else:
                    dt = min(dt_x, dt_y)
                x = min(max(x + dx * dt, 0), n - 1)
                y = min(max(y + dy * dt, 0), m - 1)
                weight = pow(np.cos(t - 0.46 * k), 2)
                backward_sum += noise[int(y), int(x)] * weight
                backward_total += weight
            result[i, j] = (forward_sum + backward_sum) / (forward_total + backward_total)
    return result


# ---------------------- notes in the visualization section ------------------------------##
#  the array is n-dimensional number array with shape (Nx, Ny) or (Nx, Ny, Nz)            ##
#  so have to align the x-axis of data with real figure x-axis (as well as y and z),      ##
#   (1) in plt.imshow section, need transpose and "origin = 'lower' "                     ##
#   (2) in cv2 section, need frame alignment as well, details shown in codes              ##
#   and due to the flexibility of cv2, can simply use ROTATE_90_COUNTERCLOCKWISE to       ##
#   achieve the same effects as plt.show section                                          ##
# ----------------------------------------------------------------------------------------##


def streamlines(Vx, Vy):
    # here the transpose is that the streamlines are based on f(x,y) while our x is the first dimension
    Vx, Vy = Vx.T, Vy.T
    x = np.linspace(0, Vx.shape[0] - 1, Vx.shape[0])
    y = np.linspace(0, Vx.shape[1] - 1, Vx.shape[1])
    x, y = np.meshgrid(y, x)
    plt.streamplot(x, y, Vx, Vy, density=0.5, color="w")


def show_color(tex, colorData=None):
    plt.figure()
    plt.rcParams["axes.facecolor"] = "white"
    if colorData is None:
        tex = tex.T
        plt.imshow(tex, origin="lower", cmap="Greys")
    else:
        tex = tex.T * 2

        colorData = colorData.T
        # normalize the color data
        colorData = (colorData - np.min(colorData)) / (np.max(colorData) - np.min(colorData))
        lic_rgba = cm.jet(colorData)
        brightness_factor = 1.5
        lic_rgba[..., 0] = np.clip(brightness_factor * tex * lic_rgba[..., 0], 0, 1)
        lic_rgba[..., 1] = np.clip(brightness_factor * tex * lic_rgba[..., 1], 0, 1)
        lic_rgba[..., 2] = np.clip(brightness_factor * tex * lic_rgba[..., 2], 0, 1)
        lic_rgba[..., 3] = 1
        lic_rgba = np.array(lic_rgba)
        plt.imshow(lic_rgba, origin="lower", cmap="jet")


def generate_animation(vector_field_x, vector_field_y, len_pix=5, output_folder="animated_lic", noise=None):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(path.join(output_folder, "images"), exist_ok=True)
    if noise is None:
        noise = get_noise(vector_field_x)
    iter_num = 5
    for t in range(100):
        lic_image = lic_2d(vector_field_x, vector_field_y, t=(t / 5), len_pix=len_pix, noise=noise)
        for _ in range(iter_num - 1):
            lic_image = lic_2d(vector_field_x, vector_field_y, t=(t / 5), len_pix=len_pix, noise=lic_image)
        # save images by plt
        show_color(lic_image, np.log10(rho))
        plt.savefig(path.join(output_folder, "images", "image_%05d.png" % t))
        plt.close()
        # save images by cv2
        # lic_image = cv2.rotate(lic_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # cv2.imwrite(path.join(output_folder, "images", "%d.png" % time5), lic_image * 255)

    # subprocess.run(["ffmpeg",
    #                 "-i", path.join(output_folder, "images", "image_%05d.png"),
    #                 "-r", "30",
    #                 "-y", path.join(output_folder, "animated_lic.mp4")])
    image_folder = output_folder + "/images/"

    def generate_video(image_folder, video_name, fps=30):
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

    generate_video(image_folder, "animated_lic.mp4")


if __name__ == "__main__":

    with h5py.File(r"D:\CUHK\Data_from_zcao\struct01\struct01_snap52.h5", "r") as f:
        B_x = f["i_mag_field"][:, :, 50]
        B_y = f["j_mag_field"][:, :, 50]
        rho = f["gas_density"][:, :, 50]

    start_time = time.time()
    # ------------------ visualization section -------------------------------
    B_x = bilinear_interpolation(B_x, 5)
    B_y = bilinear_interpolation(B_y, 5)
    rho = bilinear_interpolation(rho, 5)

    white_noise = np.random.randint(0, 10, size=B_x.shape)
    white_noise = np.where(white_noise < 7, 0, 1).astype(np.float64)  # customize the input noise
    lic_tex = lic_2d(B_x, B_y, t=0, len_pix=64, noise=white_noise)
    niter_lic = 2
    for _ in range(niter_lic - 1):
        lic_tex = lic_2d(B_x, B_y, t=0, len_pix=64, noise=lic_tex)

    print(lic_tex.shape, B_x.shape, B_y.shape)
    show_color(lic_tex, np.log10(rho))
    # streamlines(B_x, B_y)
    print("computation elapsed time: ")
    print("--- %.2f seconds ---" % (time.time() - start_time))
    plt.show()
