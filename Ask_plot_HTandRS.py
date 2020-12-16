#/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ASKfunction import Settings

# December 15, 2020: Starting Python code to implement matplotlib
# Askervein Simulation and Benchmark

settings = Settings('namelist.json')

# Setting up simulation parameters
nx = settings.nx
ny = settings.ny
nz = settings.nz
lx = settings.lx
ly = settings.ly
lz = settings.lz
X = np.linspace(0,lx,nx)
Y = np.linspace(0,ly,ny)
Z = np.linspace(0,lz,nz)

# Definind Mesh Grid
xzx, xzz = np.meshgrid(X,Z,indexing='ij')
xyx, xyy = np.meshgrid(X,Y,indexing='ij')
yzy, yzz = np.meshgrid(Y,Z,indexing='ij')

XZ = 1
XY = 2
YZ = 3

# Assigning Mean U, Mean V, and Mean W data to variables
Udata = np.loadtxt(settings.U)
Vdata = np.loadtxt(settings.V)
Wdata = np.loadtxt(settings.W)

# Ploting along desired direction
XZplane = np.zeros((Mesh[0]-1, Mesh[2]-1))
for k in range(Mesh[2]-1):
	for i in range(Mesh[0]-1);
		XZplane[i,k] = 
