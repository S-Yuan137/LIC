import numpy as np
import pyvista as pv
# from pyvista.plotting.tools import opacity_transfer_function
pv.set_jupyter_backend('pythreejs') 


def data_filter(greys, v_m,  nmin, nmax):
    nmin = (np.nanmax(v_m) - np.nanmin(v_m)) * nmin + np.nanmin(v_m)
    nmax = (np.nanmax(v_m) - np.nanmin(v_m)) * nmax + np.nanmin(v_m)
    print(np.nanmax(v_m) , np.nanmin(v_m))
    filter = (v_m >= nmin).astype(int) * (v_m <= nmax).astype(int)
    return greys * filter

vector = np.fromfile(r"D:\CUHK\VolumeLIC\VolumeLIC\tornado.dat", dtype = np.float32)

mesh = pv.UniformGrid((128,128,128), (1, 1, 1), (0,0,0))
x = mesh.points[:, 0]
y = mesh.points[:, 1]
z = mesh.points[:, 2]
vectors = np.empty((mesh.n_points, 3))
vectors[:, 0] = vector.reshape(128,128,128,3)[:,:,:,0].flatten()
vectors[:, 1] = vector.reshape(128,128,128,3)[:,:,:,1].flatten()
vectors[:, 2] = vector.reshape(128,128,128,3)[:,:,:,2].flatten()

mesh['vectors'] = vectors


p = pv.Plotter()

for i in np.arange(13):
    stream, src = mesh.streamlines('vectors', return_source=True,
                                    terminal_speed=0, n_points=10,
                                    source_radius=15,source_center=(10,10,i*10))
    p.add_mesh(stream.tube(radius=0.1), scalars="vectors", lighting=False)

p.add_mesh(mesh.outline(), color="k")



vector = np.fromfile(r"D:\CUHK\VolumeLIC\VolumeLIC/tornado.dat", dtype = np.float32)
v_x = vector.reshape(128,128,128,3)[:,:,:,0]
v_y = vector.reshape(128,128,128,3)[:,:,:,1]
v_z = vector.reshape(128,128,128,3)[:,:,:,2]

v_m = np.sqrt(v_x**2 + v_y**2 + v_z**2).flatten()
mesh['lic'] = arr = np.fromfile(r'D:\CUHK\VolumeLIC\VolumeLIC\tornado_lic.dat', dtype=np.int8)
mesh.point_data["greys"] = data_filter(arr, v_m, 0.5, 0.9) # filter the array!

p.add_mesh(mesh.outline(), color="k")
p.add_volume(mesh, cmap="Greys", opacity='linear')


p.show()