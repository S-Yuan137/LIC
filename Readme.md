# some codes for Line integral Covolution Visualziation
Author: SB

## goals
1. 2d LIC 
   1. with colour maps
   2. with parallel acceleration
2. 3d LIC 
   1. with parallel computation
   2. advanced rendering techniques
   
## possible tools
parallel: taichi, TensorFlow, Numba, 
colour: Matplotlib
visualization: PyVista 

## current state (goals achieved)
a 2d LIC using TensorFlow; a 2d LIC using Numba;
all 2d versions can be modified with colourmap inputting(for the TensorFlow one I have finished it.)

A 3d LIC using NumPy without parallel is finished. What's more, I still don't find the proper way to render the 3D LIC texture.

## current problems
1. Efficiency
2. 3D Rendering 
3. (Not fatal) 2D version can be improved

Try another method: rich streamlines 