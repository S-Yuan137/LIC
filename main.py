# from numpy.lib.function_base import meshgrid
import lic
import visualize
import numpy as np


shape = (10,10,10)
np.random.seed(1)
# ux = np.random.rand(5, 6, 7)
ux = np.ones(shape)
np.random.seed(3)
uy = np.random.randint(0,1, size = shape)
np.random.seed(10)
uz = np.random.randint(0,1, size = shape)

t = np.linspace(0, 2*np.pi, 10)
x, y, z = np.meshgrid(t,t,t, indexing = 'ij')

test_field = lic.vectorfield(shape, ux, uy, uz)

# data = lic.LIC3d(test_field, 5)
# visualize.lic_show(data)
visualize.vector_show(test_field)
visualize.streamline(test_field)