# some codes for Line integral Covolution Visualisation

Author: SB

## goals

1. 2d LIC &#10004;
   1. [x] with colour maps
   2. [x] with parallel acceleration
2. 3d LIC
   1. with parallel computation
   2. advanced rendering techniques

## possible tools

* parallel: TensorFlow, Numba, ~~taichi~~(temporarily deprecated)
* colour: Matplotlib
* visualization: PyVista (for 3D)

## current state (goals achieved)

* a 2d LIC using TensorFlow;
* a 2d LIC using Numba; *<font color=red>this is preferable</font>*
* a 2d LIC using interpolation, *without acceleration*

~~all 2d versions can be modified with colourmap inputting(for the one using the TensorFlow I have finished it.)~~ **The next step is to modify the Numba version as a python package to distribute to group members.**~~It should have (at least): 1) colourful textures 2) resample controlling~~(marked on 20220303)

A 3d LIC using NumPy without parallel is finished. What's more, I still don't find the proper way to render the 3D LIC texture.

## current problems

1. Efficiency (partly solved)

   Through experience from the 2D LIC, I found it impossible to accelerate the computation with keeping interpolation. The efficient way is to upsample the whole vector fields before convolution if one would like to improve the textures.

2. 3D Rendering

Try another method: rich streamlines
