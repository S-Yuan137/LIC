import pyvista as pv
import numpy as np
import h5py


###### extract data from hdf5 #########################
f = h5py.File('D:\CUHK\Data_from_zcao\struct03_snap38.h5', 'r')
B_i = np.array(f['i_mag_field'])
B_j = np.array(f['j_mag_field'])
B_k = np.array(f['k_mag_field'])


###### make structural data named mesh #################
mesh = pv.UniformGrid(B_i.shape, (1, 1, 1), (0,0,0))
x = mesh.points[:, 0]
y = mesh.points[:, 1]
z = mesh.points[:, 2]
vectors = np.empty((mesh.n_points, 3))
vectors[:, 0] = B_i.flatten()
vectors[:, 1] = B_j.flatten()
vectors[:, 2] = B_k.flatten()

mesh['vectors'] = vectors
B_mag = np.sqrt(B_i**2 + B_j**2 + B_k**2).flatten()
mesh['B_mag'] = B_mag


################### plot stream tubes #####################
# stream, src = mesh.streamlines('vectors', return_source=1, n_points=31,
#                                pointa= (0,0,1), pointb=(30,0,1),progress_bar=1, terminal_speed = -1)

stream, src = mesh.streamlines('vectors', return_source=1, n_points=31,
                               source_radius=10,progress_bar=1, terminal_speed = -1)
# pv.set_jupyter_backend('none')
p = pv.Plotter()
p.add_mesh(mesh.outline(), color="w")

p.add_mesh(stream, scalars="vectors", lighting=False)
p.add_mesh(src, color = 'k')
p.show_bounds(grid='front', location='outer', 
                            all_edges=True)
p.show()

