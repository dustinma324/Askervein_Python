# December 15, 2020: Starting Python code to implement matplotlib
# Askervein Simulation and Benchmark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ASKfunction import Settings, Utils, Plots

####################### Enviorment Setup #######################
utils = Utils()
plots = Plots()
settings = Settings('namelist.json')

# Setting up simulation parameters
lx = settings.lx; ly = settings.ly; lz = settings.lz
nx = settings.nxMesh2; ny = settings.nyMesh2; nz = settings.nzMesh2
X = np.linspace(0,lx,nx); Y = np.linspace(0,ly,ny); Z = np.linspace(0,lz,nz)

# Important site locations
RS = settings.RS; HT = settings.HT; CP = settings.CP

# Definind Mesh Grid
xzx, xzz = np.meshgrid(X,Z,indexing='xy')
xyx, xyy = np.meshgrid(X,Y,indexing='xy')
yzy, yzz = np.meshgrid(Y,Z,indexing='xy')

####################### Loading Data #######################
# Reading speedup lines
AALine = pd.read_csv(settings.AA, header=None)
ALine  = pd.read_csv(settings.A,  header=None)
BLine  = pd.read_csv(settings.B,  header=None)

# Creating RS and HT vertical lines
RSLine = utils.createVerticalLine(RS[0],RS[1],lz,nz*2,RS[2]-10)
HTLine = utils.createVerticalLine(HT[0],HT[1],lz,nz*2,HT[2]-10)

# Reading GIN3D simulation results
Udata = np.loadtxt(settings.U)
Vdata = np.loadtxt(settings.V)
Wdata = np.loadtxt(settings.W)
u, v, w = utils.readMesh(Udata,Vdata,Wdata,nx,ny,nz)

####################### Calculations #######################
# Calculate mean velocity magnitude 
mag = utils.calcMag(u,v,w,nx,ny,nz)

# Interpolation of non-coinciding points
AAinterp  = utils.trilinearInterpolation(mag,X,Y,Z,AALine)
Ainterp   = utils.trilinearInterpolation(mag,X,Y,Z,ALine)
Binterp   = utils.trilinearInterpolation(mag,X,Y,Z,BLine)
RSprofile = utils.trilinearInterpolation(mag,X,Y,Z,RSLine)
HTprofile = utils.trilinearInterpolation(mag,X,Y,Z,HTLine)
RS10m     = utils.trilinearInterpolation(mag,X,Y,Z,RS)

# Remove zero values and arange RS and HT by height above ground
RSprofile, RSLine = utils.convert2agl(RSprofile,RSLine,RS[2]-10.0)
HTprofile, HTLine = utils.convert2agl(HTprofile,HTLine,HT[2]-10.0)

####################### Plotting #######################
# Contour
plots.plotContourf(xyx,xyy,mag[:,:,20].T,"XY Plane","X","Y")
plots.plotContourf(xzx,xzz,mag[:,192,:].T,"XZ Plane","X","Z")

# AA, A, and B lines vs Distance to HT or CP
plots.plotFigure(AALine[0]-CP[0],(AAinterp-RS10m)/RS10m,"AA Line","Distance from CP","$\Delta$ S")
plots.plotFigure( ALine[0]-HT[0],( Ainterp-RS10m)/RS10m, "A Line","Distance from HT","$\Delta$ S")
#plots.plotFigure(,Binterp,"B Line"," ","$\Delta$ S")

# RS and HT vs Z
plots.plotSemilogy(RSprofile,RSLine[:,2],"RS","Mean Velocity ($ms^{-1}$)","$h_{agl}$ (m)")
plots.plotFigure(HTprofile,HTLine[:,2],"HT","$\Delta$ S","$h_{agl}$ (m)")

# Show all figures
plt.show()
