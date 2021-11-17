import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import numpy as np 

def vector_show(vectorfield):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    shape = vectorfield.size
    x, y, z = np.meshgrid(np.arange(0, shape[0]),
                          np.arange(0, shape[1]),
                          np.arange(0, shape[2]), indexing= 'ij')

    ax.quiver(x, y, z, vectorfield.field_x, vectorfield.field_y, vectorfield.field_z, length= 0.1)
    plt.show()

def lic_show(licdata):
    mesh = pv.UniformGrid(licdata.shape, (1, 1, 1), (0,0,0))
    mesh.point_data["greys"] = licdata.flatten() # filter the array!
    mesh.plot(show_edges=True, cmap = "Greys", opacity = 'linear')

def streamline(vectorfield):
    mesh = pv.UniformGrid(vectorfield.size, (1, 1, 1), (0,0,0))
    vectors = np.empty((mesh.n_points, 3))
    vectors[:, 0] = vectorfield.field_x.flatten()
    vectors[:, 1] = vectorfield.field_y.flatten()
    vectors[:, 2] = vectorfield.field_z.flatten()

    mesh['vectors'] = vectors
    stream, src = mesh.streamlines('vectors', return_source=True,
                                   terminal_speed=0, n_points=100,
                                   source_radius=2)


    p = pv.Plotter()
    p.add_mesh(mesh.outline(), color="k")
    p.add_mesh(stream.tube(radius=0.01), scalars="vectors", lighting=False)
    p.add_axes()


    p.show()