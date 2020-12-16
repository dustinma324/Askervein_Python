import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class Settings:
  def __init__(self,namelist):
    with open(namelist) as json_file:
      data = json.load(json_file)
    self.nx  = data["nx"]
    self.ny  = data["ny"]
    self.nz  = data["nz"]
    self.lx  = data["lx"]
    self.ly  = data["ly"]
    self.lz  = data["lz"]
    self.U   = data["UPath"]
    self.V   = data["VPath"]
    self.W   = data["WPath"]
    self.AA  = data["AALine"]
    self.A   = data["ALine"]
    self.B   = data["BLine"]

class Utils:
  # reading the dot data file to put into 3d matrix
  def readMesh(self,Udata,Vdata,Wdata,u,v,w,nx,ny,nz):

    for i in range(nx-1):
      for j in range(ny-1):
        for k in range(nz-1):
          u[i,j,k] = Udata[k*ny+j,i]
          v[i,j,k] = Vdata[k*ny+j,i]
          w[i,j,k] = Wdata[k*ny+j,i]

  # calculate the magnitude of the mean velocity field
  def calcMag(self,mag,u,v,w,nx,ny,nz):

    for i in range(nx-1):
      for j in range(ny-1):
        for k in range(nz-1):
          mag[i,j,k] = np.sqrt(u[i,j,k]**2 + v[i,j,k]**2 + w[i,j,k]**2)

  # scipy trilinear interpolation to find values along data lines
  def trilinearInterpolation(self,vel,x,y,z,line,Output):

    rgi = RegularGridInterpolator((x,y,z),vel)

    for i in range(line.shape[0]-1):
       Output[i] = rgi((line[0][i], line[1][i], line[2][i]))

