import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate

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

  # create vertical line given a specific X and Y location
  def createVerticalLine(self,x_val,y_val,lz,resolution,z0):
    array = np.ones((resolution,2))
    array[:,0] *= x_val; array[:,1] *= y_val
    tmp = np.linspace(z0,lz,resolution).reshape(resolution,1)
    array = np.concatenate((array,tmp),1)
    return array

  def removeLastElement(self,profile,line,z0):
    profile = profile[0:-2]
    line[:,2] = line[:,2]-z0
    line = line[0:-2,:]
    return profile, line

  # scipy trilinear interpolation to find values along data lines
  def trilinearInterpolation(self,vel,x,y,z,line):
    rgi = interpolate.RegularGridInterpolator((x,y,z),vel)
    return rgi((line))

  def linearInterpolation(self,z,data,line):
    f = interpolate.interp1d(z, data)
    return f((line))

class Plots:

  # lineplots
  def plotFigure(self,x,y,title,xtitle,ytitle):
    fig = plt.figure(); ax = plt.gca()
    ax.plot(x,y,"g-",label="GIN3D",linewidth=2)
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)

  # contours
  def plotContourf(self,x,y,data,title,xtitle,ytitle):
    fig = plt.figure(); ax = plt.gca()
    t = ax.contourf(x,y,data)
    fig.colorbar(t)
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)

  # HT Line
  def plotHT(self,x,y,title,xtitle,ytitle):
    fig = plt.figure(); ax = plt.gca()
    ax.plot(x,y,"r-",label="GIN3D",linewidth=2)
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)
    plt.xlim(0.0,1.6); plt.ylim(0,100)
    plt.xticks(np.arange(0,1.6+0.2,0.2)); plt.yticks(np.arange(0,100+20,20))

  # RS line
  def plotRS(self,x,y,title,xtitle,ytitle):
    yloglaw = np.linspace(0,1000,1000)
    roughloglaw = 0.654/0.41 * np.log(yloglaw/0.03)

    fig = plt.figure(); ax = plt.gca()
    ax.semilogy(roughloglaw, yloglaw,"k-", label="LogLaw",linewidth=2)
    ax.semilogy(x,y,"r-",label="GIN3D",linewidth=2)
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)
    ax.legend(loc="upper left")
    plt.xlim(0.0,20); plt.ylim(10e0,10e2)
    plt.xticks(np.arange(0,20+5,5))
