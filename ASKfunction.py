import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

class Settings:
  def __init__(self,namelist):
    with open(namelist) as json_file:
      data = json.load(json_file)
    self.nxMesh2  = data["Mesh2"]["nx"]
    self.nyMesh2  = data["Mesh2"]["ny"]
    self.nzMesh2  = data["Mesh2"]["nz"]
    self.nxMesh3  = data["Mesh3"]["nx"]
    self.nyMesh3  = data["Mesh3"]["ny"]
    self.nzMesh3  = data["Mesh3"]["nz"]
    self.lx  = data["lx"]
    self.ly  = data["ly"]
    self.lz  = data["lz"]
    self.U   = data["UPath"]
    self.V   = data["VPath"]
    self.W   = data["WPath"]
    self.AA  = data["AALine"]
    self.A   = data["ALine"]
    self.B   = data["BLine"]
    self.RS  = data["RS"]
    self.HT  = data["HT"]
    self.CP  = data["CP"]

class Utils:

  # reading the dot data file to put into 3d matrix
  def readMesh(self,Udata,Vdata,Wdata,nx,ny,nz):
    u = np.zeros((nx,ny,nz))
    v = np.zeros((nx,ny,nz))
    w = np.zeros((nx,ny,nz))

    for i in range(nx-1):
      for j in range(ny-1):
        for k in range(nz-1):
          u[i,j,k] = Udata[k*ny+j,i]
          v[i,j,k] = Vdata[k*ny+j,i]
          w[i,j,k] = Wdata[k*ny+j,i]
    return u, v, w

  # calculate the magnitude of the mean velocity field
  def calcMag(self,u,v,w,nx,ny,nz):
    mag = np.zeros((nx,ny,nz))

    for i in range(nx-1):
      for j in range(ny-1):
        for k in range(nz-1):
          mag[i,j,k] = np.sqrt(u[i,j,k]**2 + v[i,j,k]**2 + w[i,j,k]**2)
    return mag

  # scipy trilinear interpolation to find values along data lines
  def trilinearInterpolation(self,vel,x,y,z,line):

    rgi = RegularGridInterpolator((x,y,z),vel)
    return rgi((line))

  # create vertical line given a specific X and Y location
  def createVerticalLine(self,x_val,y_val,lz,resolution,z0):
    array = np.ones((resolution,2))
    array[:,0] *= x_val; array[:,1] *= y_val
    tmp = np.linspace(z0,lz,resolution).reshape(resolution,1)
    array = np.concatenate((array,tmp),1)
    return array

  def convert2agl(self,profile,line,z0):
    tmp_idx = np.argwhere(profile)
    tmp_profile = np.zeros(len(tmp_idx)-1)
    tmp_line = np.zeros((len(tmp_idx)-1,3))

    for i in range(len(tmp_idx)-1):
      tmp_profile[i] = profile[tmp_idx[i]]
      tmp_line[i] = line[tmp_idx[i],:]
    tmp_line[:,2] = tmp_line[:,2] - z0
    return tmp_profile, tmp_line

class Plots:

  # lineplots
  def plotFigure(self,x,y,title,xtitle,ytitle):

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(x,y)
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)

  # lineplots
  def plotSemilogy(self,x,y,title,xtitle,ytitle):

    fig = plt.figure()
    ax = plt.gca()
    ax.semilogy(x,y)
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)

  # contours
  def plotContourf(self,x,y,data,title,xtitle,ytitle):

    fig = plt.figure(); ax = plt.gca()
    t = ax.contourf(x,y,data)
    fig.colorbar(t)
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)
