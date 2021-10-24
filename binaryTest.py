import numpy as np
import pyvista as pv
from pyvista.plotting.tools import opacity_transfer_function
pv.set_jupyter_backend('pythreejs') 

def data_filter(greys, v_m,  nmin, nmax):
    nmin = (np.nanmax(v_m) - np.nanmin(v_m)) * nmin + np.nanmin(v_m)
    nmax = (np.nanmax(v_m) - np.nanmin(v_m)) * nmax + np.nanmin(v_m)
    print(np.nanmax(v_m) , np.nanmin(v_m))
    filter = (v_m >= nmin).astype(int) * (v_m <= nmax).astype(int)
    return greys * filter

 

arr = np.fromfile('../VolumeLIC/tornado_lic.dat', dtype=np.int8)
vector = np.fromfile("../VolumeLIC/tornado.dat", dtype = np.float32)
v_x = vector.reshape(128,128,128,3)[:,:,:,0]
v_y = vector.reshape(128,128,128,3)[:,:,:,1]
v_z = vector.reshape(128,128,128,3)[:,:,:,2]

v_m = np.sqrt(v_x**2 + v_y**2 + v_z**2).flatten('F')


# Create the spatial reference
grid = pv.UniformGrid()

# Set the grid dimensions: shape + 1 because we want to inject our values on
#   the CELL data
grid.dimensions = np.array((128,128,128))

# Edit the spatial reference
grid.origin = (0, 0, 0)  # The bottom left corner of the data set
grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis




# Add the data values to the cell data
grid.point_data["greys"] = data_filter(arr, v_m, 0.5, 0.9) # Flatten the array!


# Now plot the grid!
# grid.plot(show_edges=False)


p = pv.Plotter()
p.add_mesh(grid.outline(), color="k")
# p.add_mesh(grid['values'], opacity=opacity)
p.add_volume(grid, cmap="Greys")
p.show()

# p2 = pv.Plotter()
# smoothed_data = grid.gaussian_smooth(std_dev=.5)
# p2.add_volume(smoothed_data, cmap="Greys", opacity=[0,0,0,0.5,0.9])
# p2.show()
