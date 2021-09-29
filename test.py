import h5py
import numpy as np
from skimage.measure import marching_cubes_lewiner
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

f = h5py.File('D:\CUHK\Data_from_zcao\struct03_snap38.h5', 'r')

B_i = np.array(f['i_mag_field'])
B_j = np.array(f['j_mag_field'])
B_k = np.array(f['k_mag_field'])


B_mag = np.sqrt(B_i*B_i + B_j*B_j + B_k*B_k)
iso_val=np.mean(B_mag)
verts, faces, _, _ = marching_cubes_lewiner(B_mag, iso_val, spacing=(0.1, 0.1, 0.1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], cmap='jet',
                lw=1)
plt.show()