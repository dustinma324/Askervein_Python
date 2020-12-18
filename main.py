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

####################### Loading Field data #######################
# Important site locations
RS = settings.streamRS
HT = settings.streamHT
CP = settings.streamCP

# Reading speedup lines
AALine = pd.read_csv(settings.streamAA, header=None)
ALine  = pd.read_csv(settings.streamA,  header=None)
#BLine  = pd.read_csv(settings.streamB,  header=None)

# Reading GIN3D simulation results
Udata = np.loadtxt(settings.streamU)
Vdata = np.loadtxt(settings.streamV)
Wdata = np.loadtxt(settings.streamW)

####################### Calculations #######################
# Definind Mesh Grid
xzx, xzz = np.meshgrid(X,Z,indexing='xy')
xyx, xyy = np.meshgrid(X,Y,indexing='xy')
yzy, yzz = np.meshgrid(Y,Z,indexing='xy')

# Creating RS and HT vertical lines
RSLine = utils.createVerticalLine(RS[0],RS[1],lz,resolution,RS[2]-10)
HTLine = utils.createVerticalLine(HT[0],HT[1],lz,resolution,HT[2]-10)

# Calculate mean velocity magnitude 
u, v, w = utils.readMesh(Udata,Vdata,Wdata,nx,ny,nz)
mag = utils.calcMag(u,v,w,nx,ny,nz)

# Interpolation of non-coinciding points
AAinterp  = utils.trilinearInterpolation(mag,X,Y,Z,AALine)
Ainterp   = utils.trilinearInterpolation(mag,X,Y,Z,ALine)
#Binterp   = utils.trilinearInterpolation(mag,X,Y,Z,BLine)
RS_z      = utils.trilinearInterpolation(mag,X,Y,Z,RSLine)
HT_z      = utils.trilinearInterpolation(mag,X,Y,Z,HTLine)
RS10m     = utils.trilinearInterpolation(mag,X,Y,Z,RS)

# Finding absolute distance from CP to HT for AA, A, and B (Note: dir = 0 for x and 1 for y)
abs_AAdist = utils.findAbsDist(AALine,CP,0)
abs_Adist  = utils.findAbsDist( ALine,HT,0)
abs_Bdist  = utils.findAbsDist( BLine,HT,1)

# Remove zero values and arange RS and HT by height above ground
RS_agl, RSLine_adjusted = utils.removeLastElement(RS_z,RSLine,RS[2]-10.0)
HT_agl, HTLine_adjusted = utils.removeLastElement(HT_z,HTLine,HT[2]-10.0)
HT_normalize = utils.linearInterpolation(RSLine_adjusted[:,2],RS_agl,HTLine_adjusted[:,2])

####################### Plotting #######################
# Contour
plots.plotContourf(xyx,xyy,mag[:,:,20].T,"XY Plane","X","Y")
plots.plotContourf(xzx,xzz,mag[:,192,:].T,"XZ Plane","X","Z")

# AA, A, and B lines vs Distance to HT or CP (Normalized by constant RS10m)
plots.plotFigure(abs_AAdist,(AAinterp-RS10m)/RS10m,"AA Line","Distance from CP (m)","$\Delta$ S",[-1000,1000],[-1,1])
plots.plotFigure(abs_Adist, ( Ainterp-RS10m)/RS10m, "A Line","Distance from HT (m)","$\Delta$ S",[-1000,1000],[-1,1])
#plots.plotFigure(abs_Bdist, ( Binterp-RS10m)/RS10m, "B Line","Distance from HT (m)","$\Delta$ S",[-500,1800],[-1,1])

# RS and HT vs Z
plots.plotRS(RS_agl,RSLine_adjusted[:,2],"RS","Mean Velocity ($ms^{-1}$)","$h_{agl}$ (m)")
plots.plotHT((HT_agl-HT_normalize)/HT_normalize,HTLine_adjusted[:,2],"HT","$\Delta$ S","$h_{agl}$ (m)")

# Show all figures
plt.show()
