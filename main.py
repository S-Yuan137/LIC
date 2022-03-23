# from numpy.lib.function_base import meshgrid
#%%
import lic
import visualize
import numpy as np


shape = (20, 20, 2)
np.random.seed(1)
# ux = np.random.rand(5, 6, 7)
ux = np.ones(shape)
np.random.seed(3)
uy = np.random.randint(0, 1, size=shape)
np.random.seed(10)
uz = np.random.randint(0, 1, size=shape)


test_field = lic.vectorfield(shape, ux, ux, uz)

data = lic.LIC2d(test_field, 5)
visualize.lic_show(data)
# visualize.vector_show(test_field)
# visualize.streamline(test_field)


#%%
import lic
import visualize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

t = np.linspace(0, 100, 20)
x, y = np.meshgrid(t, t, indexing="ij")

vx = np.sin(x)
vy = np.cos(y)
lim = (0.2, 0.6)
alpha = 0.8
lic_data_rgba = cm.ScalarMappable(norm=None, cmap="jet").to_rgba(vy)
lic_data_clip_rescale = (data - lim[0]) / (lim[1] - lim[0])
lic_data_rgba[..., 3] = lic_data_clip_rescale * alpha

plt.imshow(lic_data_rgba, origin="lower", cmap="jet", alpha=alpha)
plt.show()


# %%
