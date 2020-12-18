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
resolution = nz*2

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
RSLine = utils.createVerticalLine(RS[0],RS[1],lz,resolution,RS[2]-10)
HTLine = utils.createVerticalLine(HT[0],HT[1],lz,resolution,HT[2]-10)

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
RS_z      = utils.trilinearInterpolation(mag,X,Y,Z,RSLine)
HT_z      = utils.trilinearInterpolation(mag,X,Y,Z,HTLine)
RS10m     = utils.trilinearInterpolation(mag,X,Y,Z,RS)

# Remove zero values and arange RS and HT by height above ground
RS_agl, RSLine_adjusted = utils.removeLastElement(RS_z,RSLine,RS[2]-10.0)
HT_agl, HTLine_adjusted = utils.removeLastElement(HT_z,HTLine,HT[2]-10.0)

# Interpolate agl level of HT to RS for normalization
HT_normalize = utils.linearInterpolation(RSLine_adjusted[:,2],RS_agl,HTLine_adjusted[:,2])

####################### Plotting #######################
# Contour
plots.plotContourf(xyx,xyy,mag[:,:,20].T,"XY Plane","X","Y")
plots.plotContourf(xzx,xzz,mag[:,192,:].T,"XZ Plane","X","Z")

# AA, A, and B lines vs Distance to HT or CP (Normalized by constant RS10m)
plots.plotFigure(AALine[0]-CP[0],(AAinterp-RS10m)/RS10m,"AA Line","Distance from CP","$\Delta$ S")
plots.plotFigure( ALine[0]-HT[0],( Ainterp-RS10m)/RS10m, "A Line","Distance from HT","$\Delta$ S")
#plots.plotFigure(,Binterp,"B Line"," ","$\Delta$ S")

# RS and HT vs Z
plots.plotRS(RS_agl,RSLine_adjusted[:,2],"RS","Mean Velocity ($ms^{-1}$)","$h_{agl}$ (m)")
plots.plotHT((HT_agl-HT_normalize)/HT_normalize,HTLine_adjusted[:,2],"HT","$\Delta$ S","$h_{agl}$ (m)")

# Show all figures
plt.show()
