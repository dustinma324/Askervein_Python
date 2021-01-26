import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

class Settings:
  def __init__(self,namelist):
    with open(namelist) as json_file:
      data = json.load(json_file)

    # Streamwise data
    self.SnxMesh2  = data["Streamwise"]["Mesh2"]["nx"]
    self.SnyMesh2  = data["Streamwise"]["Mesh2"]["ny"]
    self.SnzMesh2  = data["Streamwise"]["Mesh2"]["nz"]
    self.SnxMesh3  = data["Streamwise"]["Mesh3"]["nx"]
    self.SnyMesh3  = data["Streamwise"]["Mesh3"]["ny"]
    self.SnzMesh3  = data["Streamwise"]["Mesh3"]["nz"]

    self.streamlx  = data["Streamwise"]["lx"]
    self.streamly  = data["Streamwise"]["ly"]
    self.streamlz  = data["Streamwise"]["lz"]
    self.streamU   = data["Streamwise"]["GIN3DPath"]+data["Streamwise"]["Mesh3"]["MeanU"]
    self.streamV   = data["Streamwise"]["GIN3DPath"]+data["Streamwise"]["Mesh3"]["MeanV"]
    self.streamW   = data["Streamwise"]["GIN3DPath"]+data["Streamwise"]["Mesh3"]["MeanW"]
    self.streamAA  = data["Streamwise"]["AALine"]
    self.streamA   = data["Streamwise"]["ALine"]
    self.streamB   = data["Streamwise"]["BLine"]
    self.streamRS  = data["Streamwise"]["RS"]
    self.streamHT  = data["Streamwise"]["HT"]
    self.streamCP  = data["Streamwise"]["CP"]

    # Angled data
    self.AnxMesh2  = data["Angled"]["Mesh2"]["nx"]
    self.AnyMesh2  = data["Angled"]["Mesh2"]["ny"]
    self.AnzMesh2  = data["Angled"]["Mesh2"]["nz"]
    self.AnxMesh3  = data["Angled"]["Mesh3"]["nx"]
    self.AnyMesh3  = data["Angled"]["Mesh3"]["ny"]
    self.AnzMesh3  = data["Angled"]["Mesh3"]["nz"]

    self.anglelx  = data["Angled"]["lx"]
    self.anglely  = data["Angled"]["ly"]
    self.anglelz  = data["Angled"]["lz"]
    self.angleU   = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["MeanU"]
    self.angleV   = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["MeanV"]
    self.angleW   = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["MeanW"]
    self.angleAA  = data["Angled"]["AALine"]
    self.angleA   = data["Angled"]["ALine"]
    self.angleB   = data["Angled"]["BLine"]
    self.angleRS  = data["Angled"]["RS"]
    self.angleHT  = data["Angled"]["HT"]
    self.angleCP  = data["Angled"]["CP"]

    # ASK83 field data
    self.FD_AAR    = data["FieldData"]["Path"]+data["FieldData"]["AAResults"]
    self.FD_AR     = data["FieldData"]["Path"]+data["FieldData"]["AResults"]
    self.FD_BR     = data["FieldData"]["Path"]+data["FieldData"]["BResults"]
    self.FD_AAErr  = data["FieldData"]["Path"]+data["FieldData"]["AAError"]
    self.FD_AErr   = data["FieldData"]["Path"]+data["FieldData"]["AError"]
    self.FD_BErr   = data["FieldData"]["Path"]+data["FieldData"]["BError"]
    self.FD_RSKite = data["FieldData"]["Path"]+data["FieldData"]["RSKite"]
    self.FD_RSCup  = data["FieldData"]["Path"]+data["FieldData"]["RSCup"]
    self.FD_RSGill = data["FieldData"]["Path"]+data["FieldData"]["RSGill"]
    self.FD_HTR    = data["FieldData"]["Path"]+data["FieldData"]["HTField"]

    self.figurePath = data["FigurePath"]

class Utils:

  # reading field data (Do not change)
  def readField(self):
    settings = Settings('namelist.json')
    aaResults = np.array(pd.read_csv(settings.FD_AAR))
    aaErr = np.array(pd.read_csv(settings.FD_AAErr))
    aResults = np.array(pd.read_csv(settings.FD_AR))
    aErr = np.array(pd.read_csv(settings.FD_AErr))
    bResults = np.array(pd.read_csv(settings.FD_BR))
    bErr = np.array(pd.read_csv(settings.FD_BErr))
    rsKite = np.array(pd.read_csv(settings.FD_RSKite))
    rsCup = np.array(pd.read_csv(settings.FD_RSCup))
    rsGill = np.array(pd.read_csv(settings.FD_RSGill))
    htResults = np.array(pd.read_csv(settings.FD_HTR))

    # Store these arrays
    results = {
      "AAResults" : aaResults,
      "AResults"  : aResults,
      "BResults"  : bResults,
      "AAError"   : aaErr,
      "AError"    : aErr,
      "BError"    : bErr,
      "RSKite"    : rsKite,
      "RSCup"     : rsCup,
      "RSGill"    : rsGill,
      "HTResults" : htResults
    }
    return results

  # performing necessary calculation used for error bars for AA, A, and B line
  def creatingErrorBarData(self,dataField,errUp):
    dataX = dataField[:,0]*1000
    dataU = dataField[:,1]
    err = errUp[:,1] - dataField[:,1]
    return dataX, dataU, err

  # error propagation calculation for SpeedUp error (0 = RS, 1 = HT, 2 = Z)
  def errorPropagationCalc(self,data):
    DS = data[:,0]
    z  = data[:,1]
    err = data[:,2]
    return DS, z, err 

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

  # finding the absolute distance to HT and CP
  def findAbsDist(self,line,ref,dir):
    line[:][0] = line[:][0]-ref[0] #x-axis
    line[:][1] = line[:][1]-ref[1] #y-axis
    neg_idx = np.where(line[:][dir]<0)[0].max() #find last index that is a negative element
    tmpline = np.sqrt(line[:][0]**2+line[:][1]**2)
    tmpline[0:neg_idx] = tmpline[0:neg_idx] * -1
    return tmpline

  # create vertical line given a specific X and Y location
  def createVerticalLine(self,x_val,y_val,lz,resolution,z0):
    array = np.ones((resolution,2))
    array[:,0] *= x_val; array[:,1] *= y_val
    tmp = np.linspace(z0,lz,resolution).reshape(resolution,1)
    array = np.concatenate((array,tmp),1)
    return array

  # removing the last two elemtns of the passed in array and normalizing z by z0
  def removeLastElement(self,profile,line,z0):
    profile = profile[0:-2] 
    line[:,2] = line[:,2]-z0
    line = line[0:-2,:]
    return profile, line

  def cellcenter(self,u,v,w,nx,ny,nz):
    uCenter = 0.5*(u[1:,:,:]+u[:-1,:,:])
    vCenter = 0.5*(v[:,1:,:]+v[:,:-1,:])
    wCenter = 0.5*(w[:,:,1:]+w[:,:,:-1])
    return uCenter, vCenter, wCenter

class Interp:

  # scipy trilinear interpolation to find values along data lines
  def trilinearInterpolation(self,vel,x,y,z,line):
    rgi = interpolate.RegularGridInterpolator((x,y,z),vel,method='linear')
    return rgi((line))

  # scipy linear interpolation to find value along data line 
  def linearInterpolation(self,z,data,line):
    f = interpolate.interp1d(z, data)
    return f((line))

  # interpolation for a line
  def InterpolationLine(self,U,V,W,dx,dy,dz,line,size):
    line = np.array(line)
    storeVel = np.zeros((size,3))
    for i in range(size-1):
      tmp = line[i,:]
      x1 = int(np.ceil(tmp[0]/dx))
      x0 = int(x1-1)
      ax = tmp[0]/dx - x0

      y1 = int(np.ceil(tmp[1]/dy))
      y0 = int(y1-1)
      ay = tmp[1]/dy - y0

      z1 = int(np.ceil(tmp[2]/dz))
      z0 = int(z1-1)
      az = tmp[2]/dz - z0
      print(x0,x1,y0,y1,z0,z1,ax,ay,az) # 9 elements in this

      # linear interpolation of 3 components in 3 directions
      # General : y_new = (1-a) * y0 + a * y1
      storeVel[i,0] = (1.0-ax)*U[x0,y1,z1] + ax*U[x1,y1,z1]
      storeVel[i,1] = (1.0-ay)*V[x1,y0,z1] + ay*V[x1,y1,z1]
      storeVel[i,2] = (1.0-az)*W[x1,y1,z0] + az*W[x1,y1,z1]

    mag = np.sqrt(storeVel[:,0]**2+storeVel[:,1]**2+storeVel[:,2]**2)
    return mag

  # interpolation for a single point
  def InterpolationPoint(self,U,V,W,dx,dy,dz,line,size):
    line = np.array(line)
    storeVel = np.zeros(size)
    x1 = int(np.ceil(line[0]/dx))
    x0 = int(x1-1)
    ax = line[0]/dx - x0

    y1 = int(np.ceil(line[1]/dy))
    y0 = int(y1-1)
    ay = line[1]/dy - y0

    z1 = int(np.ceil(line[2]/dz))
    z0 = int(z1-1)
    az = line[2]/dz - z0
    print(x0,x1,y0,y1,z0,z1,ax,ay,az) # 9 elements in this

    # linear interpolation of 3 components in 3 directions
    storeVel[0] = (1.0-ax)*U[x0,y1,z1] + ax*U[x1,y1,z1]
    storeVel[1] = (1.0-ay)*V[x1,y0,z1] + ay*V[x1,y1,z1]
    storeVel[2] = (1.0-az)*W[x1,y1,z0] + az*W[x1,y1,z1]

    mag = np.sqrt(storeVel[0]**2+storeVel[1]**2+storeVel[2]**2)
    return mag

class Plots:

  # contours
  def plotContourf(self,x,y,data,title,xtitle,ytitle):
    fig = plt.figure(); ax = plt.gca()
    t = ax.contourf(x,y,data)
    fig.colorbar(t)
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)

  # lineplots
  def plotFigure(self,x,y,title,xtitle,ytitle,xlim,ylim,fdataname,edataname,filename):
    utils = Utils(); settings = Settings('namelist.json')

    field = utils.readField()
    dataField = field[fdataname]; errUp = field[edataname]
    dataX, dataU, err = utils.creatingErrorBarData(dataField,errUp)

    fig = plt.figure(); ax = plt.gca()
    ax.plot(x,y,"g-",label="GIN3D",linewidth=2)
    plt.errorbar(dataX,(dataU*8.9-8.9)/8.9,yerr=err,fmt='o',label="Field")
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)
    plt.xlim(xlim[0],xlim[1]); plt.ylim(ylim[0],ylim[1])

    fig.savefig(settings.figurePath+filename,dpi=1200)

  # RS line
  def plotRS(self,x,y,title,xtitle,ytitle,filename):
    utils = Utils(); settings = Settings('namelist.json')

    yloglaw = np.linspace(0,1000,1000)
    roughloglaw = 0.654/0.41 * np.log(yloglaw/0.03) # Log Law used in Figure 7 (DeLeon 2018)

    field = utils.readField()
    kite = field["RSKite"]; cup = field["RSCup"]; gill = field["RSGill"]

    fig = plt.figure(); ax = plt.gca()
    ax.semilogy(roughloglaw, yloglaw,"k-", label="LogLaw",linewidth=2)
    ax.semilogy(kite[:,1],kite[:,0],'x',color='b',label="Kite",markersize=10)
    ax.semilogy(cup[:,1],cup[:,0],'.',color='b',label="Cup",markersize=12)
    ax.semilogy(gill[:,1],gill[:,0],'^',color='b',label="Gill",markersize=10)
    ax.semilogy(x,y,"r-",label="GIN3D",linewidth=2)
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)
    ax.legend(loc="upper left")
    plt.xlim(0.0,20.0); plt.ylim(1e0,1e3)
    plt.xticks(np.arange(0,20+5,5))

    # inset plot
    ax2 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax,[0.6,0.075,0.35,0.35])
    ax2.set_axes_locator(ip)
    ax2.semilogy(roughloglaw, yloglaw,"k-", label="LogLaw",linewidth=2)
    ax2.semilogy(kite[:,1],kite[:,0],'x',color='b',label="Kite",markersize=10)
    ax2.semilogy(cup[:,1],cup[:,0],'.',color='b',label="Cup",markersize=12)
    ax2.semilogy(gill[:,1],gill[:,0],'^',color='b',label="Gill",markersize=10)
    ax2.semilogy(x,y,"r-",label="GIN3D",linewidth=2)
    ax2.set_xlim(7.0,12.0); ax2.set_ylim(3*1e0,5*1e1)

    fig.savefig(settings.figurePath+filename,dpi=1200)

  # HT line
  def plotHT(self,x,y,title,xtitle,ytitle,filename):
    utils = Utils(); settings = Settings('namelist.json')

    field = utils.readField()
    dataField = field["HTResults"]
    DS, z, err = utils.errorPropagationCalc(dataField) 

    fig = plt.figure(); ax = plt.gca()
    ax.plot(x,y,"r-",label="GIN3D",linewidth=2)
    plt.errorbar(DS,z,xerr=err,fmt='o',label="Field")
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)
    ax.legend(loc="upper right")
    plt.xlim(0.0,1.6); plt.ylim(0,100)
    plt.xticks(np.arange(0,1.6+0.2,0.2)); plt.yticks(np.arange(0,100+20,20))

    fig.savefig(settings.figurePath+filename,dpi=1200)

  def plotAlongDirection(self,x,u,v,w,nx,ny,nz):
    fig = plt.figure(); ax = plt.gca()
    ax.plot(x[:-1],u[:-1,np.round(ny/2),np.round(nz/2)],"g-",label="U",linewidth=2)
    ax.plot(x[:-1],v[:-1,np.round(ny/2),np.round(nz/2)],"k-",label="V",linewidth=2)
    ax.plot(x[:-1],w[:-1,np.round(ny/2),np.round(nz/2)],"r-",label="W",linewidth=2)
    ax.legend(loc="upper right")
