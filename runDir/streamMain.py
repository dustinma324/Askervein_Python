# December 15, 2020: Starting Python code to implement matplotlib
# Askervein Simulation and Benchmark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from askFunction import Settings, Utils, Interp, Plots

####################### Enviorment Setup #######################
utils = Utils()
interp = Interp()
plots = Plots()
settings = Settings('namelist.json')

# Setting up simulation parameters
lx = settings.streamlx; ly = settings.streamly; lz = settings.streamlz
nx = settings.SnxMesh3; ny = settings.SnyMesh3; nz = settings.SnzMesh3
dx = lx/(nx-2); dy = ly/(ny-2); dz = lz/(nz-2)
X = np.linspace(0,lx,nx); Y = np.linspace(0,ly,ny); Z = np.linspace(0,lz,nz)
resolution = nz

####################### Loading Field data #######################
# Important site locations
RS = np.array(settings.streamRS)
HT = np.array(settings.streamHT)
CP = np.array(settings.streamCP)
offset = 24.7-RS[2]

# Reading speedup lines
AALine = pd.read_csv(settings.streamAA, header=None)
ALine  = pd.read_csv(settings.streamA,  header=None)
BLine  = pd.read_csv(settings.streamB,  header=None)
AALine[:][2] = AALine[:][2]-offset
ALine[:][2]  =  ALine[:][2]-offset

# Reading GIN3D simulation results
Udata = np.loadtxt(settings.streamU)
Vdata = np.loadtxt(settings.streamV)
Wdata = np.loadtxt(settings.streamW)

####################### Calculations #######################
# Creating RS and HT vertical lines
RSLine = utils.createVerticalLine(RS[0],RS[1],lz-0.5*dz,resolution-1,RS[2]-10)
HTLine = utils.createVerticalLine(HT[0],HT[1],lz-0.5*dz,resolution-1,HT[2]-10)

# Calculate mean velocity magnitude 
u, v, w = utils.readMesh(Udata,Vdata,Wdata,nx,ny,nz)
u, v, w, X, Y, Z = utils.cellcenter(u,v,w,X,Y,Z,dx,dy,dz)

# Interpolation of non-coinciding points
if 0:
  AAinterp = interp.InterpolationLine(u,v,w,dx,dy,dz,AALine,AALine.shape[0])
  Ainterp  = interp.InterpolationLine(u,v,w,dx,dy,dz,ALine,ALine.shape[0])
  Binterp  = interp.InterpolationLine(u,v,w,dx,dy,dz,BLine,BLine.shape[0])
  RS_z     = interp.InterpolationLine(u,v,w,dx,dy,dz,RSLine,RSLine.shape[0])
  HT_z     = interp.InterpolationLine(u,v,w,dx,dy,dz,HTLine,HTLine.shape[0])
  RS10m    = interp.InterpolationPoint(u,v,w,dx,dy,dz,RS,RS.shape[0])

if 1:
  mag = utils.calcMag(u,v,w,nx-1,ny-1,nz-1)
  AAinterp  = interp.trilinearInterpolation(mag,X,Y,Z,AALine)
  Ainterp   = interp.trilinearInterpolation(mag,X,Y,Z,ALine)
  Binterp   = interp.trilinearInterpolation(mag,X,Y,Z,BLine)
  RS_z      = interp.trilinearInterpolation(mag,X,Y,Z,RSLine)
  HT_z      = interp.trilinearInterpolation(mag,X,Y,Z,HTLine)
  RS10m     = interp.trilinearInterpolation(mag,X,Y,Z,RS)

####################### Plotting #######################
# Contour
if 0:
  xzx, xzz = np.meshgrid(X,Z,indexing='ij')
  xyx, xyy = np.meshgrid(X,Y,indexing='ij')
  yzy, yzz = np.meshgrid(Y,Z,indexing='ij')
  plots.plotContourf(xyx,xyy,mag[:,:,20],"XY Plane","X","Y")
  plots.plotContourf(xzx,xzz,mag[:,int(np.floor(ny/2)),:],"XZ Plane","X","Z")
  plots.plotContourf(yzy,yzz,mag[int(np.floor(nx/2)),:,:],"YZ Plane","Y","Z")

# AA, A, and B lines vs Distance to HT or CP (Normalized by constant RS10m)
if 1:
# Finding absolute distance from CP to HT for AA, A, and B (Note: dir = 0 for x and 1 for y)
  abs_AAdist = utils.findAbsDist(AALine,CP,0)
  abs_Adist  = utils.findAbsDist( ALine,HT,0)
  abs_Bdist  = utils.findAbsDist( BLine,HT,1)

  plots.plotFigure(abs_AAdist,(AAinterp-RS10m)/RS10m,"AA Line","Distance from CP (m)","$\Delta$ S",[-1000,1000],[-1,1],"AAResults","AAError","AALine.png")
  plots.plotFigure(abs_Adist, ( Ainterp-RS10m)/RS10m, "A Line","Distance from HT (m)","$\Delta$ S",[-1000,1000],[-1,1],"AResults","AError","ALine.png")
#plots.plotFigure(abs_Bdist, ( Binterp-RS10m)/RS10m, "B Line","Distance from HT (m)","$\Delta$ S",[-500,1800],[-1,1],"BResults","BError","BLine.png")

# RS and HT vs Z
if 1:
  # Remove last two elements and arange RS and HT by height above ground
  RS_agl, RSLine_adjusted = utils.removeLastElement(RS_z,RSLine,RS[2]-10.0)
  HT_agl, HTLine_adjusted = utils.removeLastElement(HT_z,HTLine,HT[2]-10.0)
  # Linear interpolation of U_RS(Z') using Z_HT, for normalization of HT profile vs agl
  HT_normalize = interp.linearInterpolation(RSLine_adjusted[:,2],RS_agl,HTLine_adjusted[:,2])

  plots.plotRS(RS_agl,RSLine_adjusted[:,2],"RS","Mean Velocity ($ms^{-1}$)","$h_{agl}$ (m)","RSloglaw.png")
  plots.plotHT((HT_agl-HT_normalize)/HT_normalize,HTLine_adjusted[:,2],"HT","$\Delta$ S","$h_{agl}$ (m)","HTnormalized.png")
  plots.plotLine(HT_z[:-1],HTLine[:-1,2],"HT","Mean Velocity","$h_{agl}$ (m)")

# Show all figures
plt.show()
