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
D_lx = settings.anglelx; D_ly = settings.anglely; D_lz = settings.anglelz
D_nx = settings.AnxMesh3; D_ny = settings.AnyMesh3; D_nz = settings.AnzMesh3
D_dx = D_lx/(D_nx-2); D_dy = D_ly/(D_ny-2); D_dz = D_lz/(D_nz-2)
D_X = np.linspace(0,D_lx,D_nx); D_Y = np.linspace(0,D_ly,D_ny); D_Z = np.linspace(0,D_lz,D_nz)

####################### Loading Field data #######################
# Important site locations
D_RS = np.array(settings.angleRS)
D_HT = np.array(settings.angleHT)
D_CP = np.array(settings.angleCP)

# Reading speedup lines
D_AALine = pd.read_csv(settings.angleAA, header=None)
D_ALine  = pd.read_csv(settings.angleA,  header=None)
D_AALine = utils.adjustLines(D_AALine,-1.0)
D_ALine = utils.adjustLines(D_ALine,-1.0)

# Reading GIN3D simulation results
D_Udata = np.loadtxt(settings.dirichletU)
D_Vdata = np.loadtxt(settings.dirichletV)
D_Wdata = np.loadtxt(settings.dirichletW)

####################### Calculations #######################
# Creating RS and HT vertical lines
D_RSLine = utils.createVerticalLine(D_RS[0],D_RS[1],D_RS[2]-10,D_Z)
D_HTLine = utils.createVerticalLine(D_HT[0],D_HT[1],D_HT[2]-10,D_Z)

# Calculate mean velocity magnitude
D_u, D_v, D_w = utils.readMesh(D_Udata,D_Vdata,D_Wdata,D_nx,D_ny,D_nz)
D_u, D_v, D_w, D_X, D_Y, D_Z = utils.cellcenter(D_u,D_v,D_w,D_X,D_Y,D_Z,D_dx,D_dy,D_dz)

# getting ratio line at different heights
D_theta = utils.calcTheta(D_u,D_v,D_w,D_nx-1,D_ny-1,D_nz-1)
D_ALine_50m   = utils.adjustLines(D_ALine,40.0)
D_ALine_100m  = utils.adjustLines(D_ALine,90.0)
D_AALine_50m  = utils.adjustLines(D_AALine,40.0)
D_AALine_100m = utils.adjustLines(D_AALine,90.0)

D_Atheta_10m    = interp.trilinearInterpolation(D_theta,D_X,D_Y,D_Z,D_ALine)
D_Atheta_50m    = interp.trilinearInterpolation(D_theta,D_X,D_Y,D_Z,D_ALine_50m)
D_Atheta_100m   = interp.trilinearInterpolation(D_theta,D_X,D_Y,D_Z,D_ALine_100m)
D_AAtheta_10m   = interp.trilinearInterpolation(D_theta,D_X,D_Y,D_Z,D_AALine)
D_AAtheta_50m   = interp.trilinearInterpolation(D_theta,D_X,D_Y,D_Z,D_AALine_50m)
D_AAtheta_100m = interp.trilinearInterpolation(D_theta,D_X,D_Y,D_Z,D_AALine_100m)

################################### ANGLED #####################################
# Setting up simulation parameters
N_lx = settings.anglelx; N_ly = settings.anglely; N_lz = settings.anglelz
N_nx = settings.AnxMesh3; N_ny = settings.AnyMesh3; N_nz = settings.AnzMesh3
N_dx = N_lx/(N_nx-2); N_dy = N_ly/(N_ny-2); N_dz = N_lz/(N_nz-2)
N_X = np.linspace(0,N_lx,N_nx); N_Y = np.linspace(0,N_ly,N_ny); N_Z = np.linspace(0,N_lz,N_nz)

####################### Loading Field data #######################
# Important site locations
N_RS = np.array(settings.angleRS)
N_HT = np.array(settings.angleHT)
N_CP = np.array(settings.angleCP)

# Reading speedup lines
N_AALine = pd.read_csv(settings.angleAA, header=None)
N_ALine  = pd.read_csv(settings.angleA,  header=None)
N_AALine = utils.adjustLines(N_AALine,-1.0)
N_ALine = utils.adjustLines(N_ALine,-1.0)

# Reading GIN3D simulation results
N_Udata = np.loadtxt(settings.angleU3)
N_Vdata = np.loadtxt(settings.angleV3)
N_Wdata = np.loadtxt(settings.angleW3)

####################### Calculations #######################
# Calculate mean velocity magnitude 
N_u, N_v, N_w = utils.readMesh(N_Udata,N_Vdata,N_Wdata,N_nx,N_ny,N_nz)
N_u, N_v, N_w, N_X, N_Y, N_Z = utils.cellcenter(N_u,N_v,N_w,N_X,N_Y,N_Z,N_dx,N_dy,N_dz)

# getting ratio line at different heights
N_theta = utils.calcTheta(N_u,N_v,N_w,N_nx-1,N_ny-1,N_nz-1)
N_ALine_50m   = utils.adjustLines(N_ALine,40.0)
N_ALine_100m  = utils.adjustLines(N_ALine,90.0)
N_AALine_50m  = utils.adjustLines(N_AALine,40.0)
N_AALine_100m = utils.adjustLines(N_AALine,90.0)

N_Atheta_10m    = interp.trilinearInterpolation(N_theta,N_X,N_Y,N_Z,N_ALine)
N_Atheta_50m    = interp.trilinearInterpolation(N_theta,N_X,N_Y,N_Z,N_ALine_50m)
N_Atheta_100m   = interp.trilinearInterpolation(N_theta,N_X,N_Y,N_Z,N_ALine_100m)
N_AAtheta_10m   = interp.trilinearInterpolation(N_theta,N_X,N_Y,N_Z,N_AALine)
N_AAtheta_50m   = interp.trilinearInterpolation(N_theta,N_X,N_Y,N_Z,N_AALine_50m)
N_AAtheta_100m = interp.trilinearInterpolation(N_theta,N_X,N_Y,N_Z,N_AALine_100m)

############################### PLOTTING ##############################
# AA, A, and B lines vs Distance to HT or CP (Normalized by constant RS10m)
D_abs_AAdist = utils.findAbsDistX(D_AALine,D_CP)
D_abs_Adist  = utils.findAbsDistX( D_ALine,D_HT)
N_abs_AAdist = utils.findAbsDist(N_AALine)
N_abs_Adist  = utils.findAbsDist( N_ALine)

plots.plotRatioBoth(D_abs_AAdist,D_AAtheta_10m,D_AAtheta_50m,D_AAtheta_100m,N_abs_AAdist,N_AAtheta_10m,N_AAtheta_50m,N_AAtheta_100m,"Distance from CP (m)","(tan$^{-1}$(V/U)-rad(60))/rad(60)",[-1000,1000],[-0.1,0.6],"AALineRatio.eps")
plots.plotRatioBoth(D_abs_Adist,D_Atheta_10m,D_Atheta_50m,D_Atheta_100m,N_abs_Adist,N_Atheta_10m,N_Atheta_50m,N_Atheta_100m,"Distance from HT (m)","(tan$^{-1}$(V/U)-rad(60))/rad(60)",[-1000,1000],[-0.1,0.6],"ALineRatio.eps")

# Show all figures
plt.show()
