# December 15, 2020: Starting Python code to implement matplotlib
# Askervein Simulation and Benchmark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ASKfunction import Settings, Utils

# Enviorment Setup
utils = Utils()
settings = Settings('namelist.json')

# Import plot lines and simulation data 
AALine = pd.read_csv(settings.AA, header=None)
ALine  = pd.read_csv(settings.A,  header=None)
BLine  = pd.read_csv(settings.B,  header=None)
Udata = np.loadtxt(settings.U)
Vdata = np.loadtxt(settings.V)
Wdata = np.loadtxt(settings.W)

# Need to include RS and HT vertical profile line
# Setting up simulation parameters
nx = settings.nx; ny = settings.ny; nz = settings.nz
lx = settings.lx; ly = settings.ly; lz = settings.lz
X = np.linspace(0,lx,nx)
Y = np.linspace(0,ly,ny)
Z = np.linspace(0,lz,nz)

# Definind Mesh Grid (Figure out how to use this with contourf)
xzx, xzz = np.meshgrid(X,Z,indexing='xy')
xyx, xyy = np.meshgrid(X,Y,indexing='xy')
yzy, yzz = np.meshgrid(Y,Z,indexing='xy')

# Defining matrix size for important mean variables
u = np.zeros((nx,ny,nz))
v = np.zeros((nx,ny,nz))
w = np.zeros((nx,ny,nz))
mag = np.zeros((nx,ny,nz))

# Reading Mesh
utils.readMesh(Udata,Vdata,Wdata,u,v,w,nx,ny,nz)

# Calculate mean magnitude
utils.calcMag(mag,u,v,w,nx,ny,nz)

# Plotting contour
#fig1 = plt.contourf(mag[:,:,20].T)
#plt.show(fig1)
#fig2 = plt.contourf(mag[:,192,:].T)
#plt.show(fig2)

# Interpolation of non-coinciding points
AAinterp = np.zeros(len(AALine))
Ainterp  = np.zeros(len(ALine))
Binterp  = np.zeros(len(BLine))
utils.trilinearInterpolation(mag,X,Y,Z,AALine,AAinterp)
utils.trilinearInterpolation(mag,X,Y,Z,ALine,Ainterp)
utils.trilinearInterpolation(mag,X,Y,Z,BLine,Binterp)
