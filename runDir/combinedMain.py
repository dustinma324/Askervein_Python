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

################################# STREAMWISE #################################
# Setting up simulation parameters
S_lx = settings.streamlx; S_ly = settings.streamly; S_lz = settings.streamlz
S_nx = settings.SnxMesh3; S_ny = settings.SnyMesh3; S_nz = settings.SnzMesh3
S_dx = S_lx/(S_nx-2); S_dy = S_ly/(S_ny-2); S_dz = S_lz/(S_nz-2)
S_X = np.linspace(0,S_lx,S_nx); S_Y = np.linspace(0,S_ly,S_ny); S_Z = np.linspace(0,S_lz,S_nz)

####################### Loading Field data #######################
# Important site locations
S_RS = np.array(settings.streamRS)
S_HT = np.array(settings.streamHT)
S_CP = np.array(settings.streamCP)

# Reading speedup lines
S_AALine = pd.read_csv(settings.streamAA, header=None)
S_ALine  = pd.read_csv(settings.streamA,  header=None)

# Reading GIN3D simulation results
S_Udata = np.loadtxt(settings.streamU3)
S_Vdata = np.loadtxt(settings.streamV3)
S_Wdata = np.loadtxt(settings.streamW3)

####################### Calculations #######################
# Creating RS and HT vertical lines
S_RSLine = utils.createVerticalLine(S_RS[0],S_RS[1],S_RS[2]-10,S_Z)
S_HTLine = utils.createVerticalLine(S_HT[0],S_HT[1],S_HT[2]-10,S_Z)

# Calculate mean velocity magnitude
S_u, S_v, S_w = utils.readMesh(S_Udata,S_Vdata,S_Wdata,S_nx,S_ny,S_nz)
S_u, S_v, S_w, S_X, S_Y, S_Z = utils.cellcenter(S_u,S_v,S_w,S_X,S_Y,S_Z,S_dx,S_dy,S_dz)

# Interpolation of non-coinciding points
S_mag = utils.calcMag(S_u,S_v,S_w,S_nx-1,S_ny-1,S_nz-1)
S_AAinterp  = interp.trilinearInterpolation(S_mag,S_X,S_Y,S_Z,S_AALine)
S_Ainterp   = interp.trilinearInterpolation(S_mag,S_X,S_Y,S_Z,S_ALine)
S_RS_z      = interp.trilinearInterpolation(S_mag,S_X,S_Y,S_Z,S_RSLine)
S_HT_z      = interp.trilinearInterpolation(S_mag,S_X,S_Y,S_Z,S_HTLine)
S_RS10m     = interp.trilinearInterpolation(S_mag,S_X,S_Y,S_Z,S_RS)

################################### ANGLED #####################################
# Setting up simulation parameters
A_lx = settings.anglelx; A_ly = settings.anglely; A_lz = settings.anglelz
A_nx = settings.AnxMesh3; A_ny = settings.AnyMesh3; A_nz = settings.AnzMesh3
A_dx = A_lx/(A_nx-2); A_dy = A_ly/(A_ny-2); A_dz = A_lz/(A_nz-2)
A_X = np.linspace(0,A_lx,A_nx); A_Y = np.linspace(0,A_ly,A_ny); A_Z = np.linspace(0,A_lz,A_nz)

####################### Loading Field data #######################
# Important site locations
A_RS = np.array(settings.angleRS)
A_HT = np.array(settings.angleHT)
A_CP = np.array(settings.angleCP)

# Reading speedup lines
A_AALine = pd.read_csv(settings.angleAA, header=None)
A_ALine  = pd.read_csv(settings.angleA,  header=None)

# Reading GIN3D simulation results
A_Udata = np.loadtxt(settings.angleU3)
A_Vdata = np.loadtxt(settings.angleV3)
A_Wdata = np.loadtxt(settings.angleW3)

####################### Calculations #######################
# Creating RS and HT vertical lines
A_RSLine = utils.createVerticalLine(A_RS[0],A_RS[1],A_RS[2]-10,A_Z)
A_HTLine = utils.createVerticalLine(A_HT[0],A_HT[1],A_HT[2]-10,A_Z)

# Calculate mean velocity magnitude 
A_u, A_v, A_w = utils.readMesh(A_Udata,A_Vdata,A_Wdata,A_nx,A_ny,A_nz)
A_u, A_v, A_w, A_X, A_Y, A_Z = utils.cellcenter(A_u,A_v,A_w,A_X,A_Y,A_Z,A_dx,A_dy,A_dz)

# Interpolation of non-coinciding points
A_mag = utils.calcMag(A_u,A_v,A_w,A_nx-1,A_ny-1,A_nz-1)
A_AAinterp  = interp.trilinearInterpolation(A_mag,A_X,A_Y,A_Z,A_AALine)
A_Ainterp   = interp.trilinearInterpolation(A_mag,A_X,A_Y,A_Z,A_ALine)
A_RS_z      = interp.trilinearInterpolation(A_mag,A_X,A_Y,A_Z,A_RSLine)
A_HT_z      = interp.trilinearInterpolation(A_mag,A_X,A_Y,A_Z,A_HTLine)
A_RS10m     = interp.trilinearInterpolation(A_mag,A_X,A_Y,A_Z,A_RS)

############################### PLOTTING ##############################
# AA, A, and B lines vs Distance to HT or CP (Normalized by constant RS10m)
S_abs_AAdist = utils.findAbsDistX(S_AALine,S_CP)
S_abs_Adist  = utils.findAbsDistX( S_ALine,S_HT)
A_abs_AAdist = utils.findAbsDist(A_AALine)
A_abs_Adist  = utils.findAbsDist( A_ALine)

plots.plotFigureBoth(S_abs_AAdist,(S_AAinterp-S_RS10m)/S_RS10m,A_abs_AAdist,(A_AAinterp-A_RS10m)/A_RS10m,"AA Line","Distance from CP (m)","$\Delta$ S",[-1000,1000],[-1,1],"AAResults","AAError","AALine.png")
plots.plotFigureBoth(S_abs_Adist, ( S_Ainterp-S_RS10m)/S_RS10m,A_abs_Adist, ( A_Ainterp-A_RS10m)/A_RS10m, "A Line","Distance from HT (m)","$\Delta$ S",[-1000,1000],[-1,1],"AResults","AError","ALine.png")

# RS and HT vs Z
# Remove last two elements and arange RS and HT by height above ground
S_RS_agl, S_RSLine_adjusted = utils.removeLastElement(S_RS_z,S_RSLine,S_RS[2]-10.0)
S_HT_agl, S_HTLine_adjusted = utils.removeLastElement(S_HT_z,S_HTLine,S_HT[2]-10.0)
A_RS_agl, A_RSLine_adjusted = utils.removeLastElement(A_RS_z,A_RSLine,A_RS[2]-10.0)
A_HT_agl, A_HTLine_adjusted = utils.removeLastElement(A_HT_z,A_HTLine,A_HT[2]-10.0)

# Linear interpolation of U_RS(Z') using Z_HT, for normalization of HT profile vs agl
S_HT_normalize = interp.linearInterpolation(S_RSLine_adjusted[:,2],S_RS_agl,S_HTLine_adjusted[:,2])
A_HT_normalize = interp.linearInterpolation(A_RSLine_adjusted[:,2],A_RS_agl,A_HTLine_adjusted[:,2])

plots.plotRSBoth(S_RS_agl,S_RSLine_adjusted[:,2],A_RS_agl,A_RSLine_adjusted[:,2],"RS","Mean Velocity ($ms^{-1}$)","$h_{agl}$ (m)","RSloglaw.png")
plots.plotHTBoth((S_HT_agl-S_HT_normalize)/S_HT_normalize,S_HTLine_adjusted[:,2],(A_HT_agl-A_HT_normalize)/A_HT_normalize,A_HTLine_adjusted[:,2],"HT","$\Delta$ S","$h_{agl}$ (m)","HTnormalized.png")

# Show all figures
plt.show()
