import numpy as np
import h5py
from numba import njit
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit, prange
import os
import os.path as path
import cv2
import subprocess
import shutil


def get_noise(field_component):
    return np.random.rand(*(field_component.shape))

@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def bilinear_interpolation(f_in, resampleFactor):
    x_in = np.linspace(0, f_in.shape[0]-1, f_in.shape[0])
    y_in = np.linspace(0, f_in.shape[1]-1, f_in.shape[1])
    x_out = np.linspace(0, f_in.shape[0]-1, f_in.shape[0]*resampleFactor)
    y_out = np.linspace(0, f_in.shape[1]-1, f_in.shape[1]*resampleFactor)

    f_out = np.zeros((y_out.size, x_out.size))
    
    for i in prange(f_out.shape[1]):
        idx = np.searchsorted(x_in, x_out[i])
        
        x1 = x_in[idx-1]
        x2 = x_in[idx]
        x = x_out[i]
        
        for j in prange(f_out.shape[0]):
            idy = np.searchsorted(y_in, y_out[j])
            y1 = y_in[idy-1]
            y2 = y_in[idy]
            y = y_out[j]

            
            f11 = f_in[idy-1, idx-1]
            f21 = f_in[idy-1, idx]
            f12 = f_in[idy, idx-1]
            f22 = f_in[idy, idx]
            

            
            f_out[j, i] = ((f11 * (x2 - x) * (y2 - y) +
                            f21 * (x - x1) * (y2 - y) +
                            f12 * (x2 - x) * (y - y1) +
                            f22 * (x - x1) * (y - y1)) /
                           ((x2 - x1) * (y2 - y1)))
    
    return f_out


@njit
def lic_2d(vector_field_x, vector_field_y, t=0, len_pix=5, noise=None):
    # here the meaning of noise as an arguement is that one can customize noise dispite of the compatibility of numba
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


##---------------------- notes in the visualization section ------------------------------##
## the array is n-dimensional number array with shape (Nx, Ny) or (Nx, Ny, Nz)            ##
## so have to align the x-axis of data with real figure x-axis (as well as y and z),      ##
##  (1) in plt.show section, need transpose and "origin = 'lower' "                       ##
##  (2) in cv2 section, need frame alignment as well, details shown in codes              ##
##  and due to the flexibility of cv2, can simply use ROTATE_90_COUNTERCLOCKWISE to       ##
##  achieve the same effects as plt.show section                                          ##
##----------------------------------------------------------------------------------------##

def streamlines(Vx, Vy):
    Vx, Vy = Vx.T, Vy.T
    x = np.linspace(0, Vx.shape[0]-1, Vx.shape[0])
    y = np.linspace(0, Vx.shape[1]-1, Vx.shape[1])
    x, y = np.meshgrid(y, x)
    print(x.shape, y.shape, Vx.shape)

    plt.streamplot(x, y, Vx, Vy, density=1)

def show_color(tex, colorData = None):
    plt.figure()
    plt.rcParams['axes.facecolor'] = 'white'
    if colorData is None:
        tex = tex.T
        plt.imshow(tex, origin = 'lower', cmap='Greys')
    else:
        tex = tex.T
        colorData = colorData.T
        lim = (0.2,0.8)
        lic_data_clip = np.clip(tex,lim[0],lim[1])
        alpha = 0.8
        lic_data_rgba = cm.ScalarMappable(norm=None,cmap='jet').to_rgba(colorData)
        lic_data_clip_rescale = (lic_data_clip - lim[0]) / (lim[1] - lim[0])
        lic_data_rgba[...,3] = lic_data_clip_rescale * alpha 
        plt.imshow(lic_data_rgba, origin='lower')

def generate_animation(vector_field_x, vector_field_y, len_pix=5, output_folder = "animated_lic", noise=None):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(path.join(output_folder, "images"), exist_ok=True)
    if noise is None:
        noise = get_noise(vector_field_x)
    for t in range(100):
        lic_image = lic_2d(vector_field_x, vector_field_y, t=(t/5), len_pix=20, noise=noise)
        lic_image = cv2.rotate(lic_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(path.join(output_folder, "images", "%d.png" % t), lic_image * 255)

    subprocess.run(["ffmpeg",
                    "-i", path.join(output_folder, "images", "%d.png"),
                    "-r", "30",
                    "-y", path.join(output_folder, "animated_lic.mp4")])
    

if __name__ == "__main__":

    with h5py.File("D:\CUHK\Data_from_zcao\struct01\struct01_snap52.h5", 'r') as f:
        B_x = f['i_mag_field'][:,:,50]
        B_y = f['j_mag_field'][:,:,50]
        rho = f['gas_density'][:,:,50]
    start_time = time.time()
    B_x = bilinear_interpolation(B_x, 3)
    B_y = bilinear_interpolation(B_y, 3)
    rho = bilinear_interpolation(rho, 3)

    white_noise = np.random.randint(0 ,10, size = B_x.shape)
    white_noise = np.where(white_noise<7, 0 , 1).astype(np.float64) # customize the input noise
    lic_tex = lic_2d(B_x, B_y, t = 0, len_pix=50, noise = white_noise)
    
    show_color(lic_tex, rho)
    

    print("computation elapsed time: ")
    print("--- %.2f seconds ---" % (time.time() - start_time))
    plt.show()
    

    
    
    



