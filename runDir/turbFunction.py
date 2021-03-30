import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import interpolate
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
import matplotlib.font_manager

class Settings:
  def __init__(self,namelist):
    with open(namelist) as json_file:
      data = json.load(json_file)

    # Streamwise data
    self.streamlx  = data["Streamwise"]["lx"]
    self.streamly  = data["Streamwise"]["ly"]
    self.streamlz  = data["Streamwise"]["lz"]
    self.streamAA  = data["Streamwise"]["AALine"]
    self.streamA   = data["Streamwise"]["ALine"]
    self.streamB   = data["Streamwise"]["BLine"]
    self.streamRS  = data["Streamwise"]["RS"]
    self.streamHT  = data["Streamwise"]["HT"]
    self.streamCP  = data["Streamwise"]["CP"]

    self.SnxMesh3  = data["Streamwise"]["Mesh3"]["nx"]
    self.SnyMesh3  = data["Streamwise"]["Mesh3"]["ny"]
    self.SnzMesh3  = data["Streamwise"]["Mesh3"]["nz"]

    self.streamInstU3   = data["Streamwise"]["GIN3DPath"]+data["Streamwise"]["Mesh3"]["InstU"]
    self.streamInstV3   = data["Streamwise"]["GIN3DPath"]+data["Streamwise"]["Mesh3"]["InstV"]
    self.streamInstW3   = data["Streamwise"]["GIN3DPath"]+data["Streamwise"]["Mesh3"]["InstW"]

    self.streamMeanU3   = data["Streamwise"]["GIN3DPath"]+data["Streamwise"]["Mesh3"]["MeanU"]
    self.streamMeanV3   = data["Streamwise"]["GIN3DPath"]+data["Streamwise"]["Mesh3"]["MeanV"]
    self.streamMeanW3   = data["Streamwise"]["GIN3DPath"]+data["Streamwise"]["Mesh3"]["MeanW"]


    # Angled (Neumann) data
    self.anglelx  = data["Angled"]["lx"]
    self.anglely  = data["Angled"]["ly"]
    self.anglelz  = data["Angled"]["lz"]
    self.angleAA  = data["Angled"]["AALine"]
    self.angleA   = data["Angled"]["ALine"]
    self.angleB   = data["Angled"]["BLine"]
    self.angleRS  = data["Angled"]["RS"]
    self.angleHT  = data["Angled"]["HT"]
    self.angleCP  = data["Angled"]["CP"]

    self.AnxMesh3  = data["Angled"]["Mesh3"]["nx"]
    self.AnyMesh3  = data["Angled"]["Mesh3"]["ny"]
    self.AnzMesh3  = data["Angled"]["Mesh3"]["nz"]

    self.angleInstU3   = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["InstU"]
    self.angleInstV3   = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["InstV"]
    self.angleInstW3   = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["InstW"]

    self.angleMeanU3   = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["MeanU"]
    self.angleMeanV3   = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["MeanV"]
    self.angleMeanW3   = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["MeanW"]

    self.angleUU = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["uu"]
    self.angleVV = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["vv"]
    self.angleWW = data["Angled"]["GIN3DPath"]+data["Angled"]["Mesh3"]["ww"]

    # Angled (Dirichlet) data
    self.dirichletInstU = data["Dirichlet"]["GIN3DPath"]+data["Dirichlet"]["Mesh"]["InstU"]
    self.dirichletInstV = data["Dirichlet"]["GIN3DPath"]+data["Dirichlet"]["Mesh"]["InstV"]
    self.dirichletInstW = data["Dirichlet"]["GIN3DPath"]+data["Dirichlet"]["Mesh"]["InstW"]

    self.dirichletMeanU = data["Dirichlet"]["GIN3DPath"]+data["Dirichlet"]["Mesh"]["MeanU"]
    self.dirichletMeanV = data["Dirichlet"]["GIN3DPath"]+data["Dirichlet"]["Mesh"]["MeanV"]
    self.dirichletMeanW = data["Dirichlet"]["GIN3DPath"]+data["Dirichlet"]["Mesh"]["MeanW"]

    self.figurePath = data["FigurePath"]

class Utils:


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

  # calculate the Turbulent intensity at RS and HT
  def calcTI(self,u,v,w,uInst,vInst,wInst,nx,ny,nz,x,y,z,line):
    interp = Interp();
    Vel = np.zeros((line.shape[0],6))
    Vel[:,0] = interp.trilinearInterpolation(uInst,x,y,z,line)
    Vel[:,1] = interp.trilinearInterpolation(vInst,x,y,z,line)
    Vel[:,2] = interp.trilinearInterpolation(wInst,x,y,z,line)
    Vel[:,3] = interp.trilinearInterpolation(u,x,y,z,line)
    Vel[:,4] = interp.trilinearInterpolation(v,x,y,z,line)
    Vel[:,5] = interp.trilinearInterpolation(w,x,y,z,line)

    TI = np.sqrt(((Vel[:,0]-Vel[:,3])**2+(Vel[:,1]-Vel[:,4])**2+(Vel[:,2]-Vel[:,5])**2)/3)/np.sqrt(Vel[:,3]**2+Vel[:,4]**2+Vel[:,5]**2)

    return TI

  # calculate the Turbulent intensity based on longitudinal direction
  def calcTILong(self,u,v,uInst,vInst,x,y,z,line,theta):
    interp = Interp();
    Vel = np.zeros((line.shape[0],4))
    Vel[:,0] = interp.trilinearInterpolation(uInst,x,y,z,line)
    Vel[:,1] = interp.trilinearInterpolation(vInst,x,y,z,line)
    Vel[:,2] = interp.trilinearInterpolation(u,x,y,z,line)
    Vel[:,3] = interp.trilinearInterpolation(v,x,y,z,line)

    InstLong = Vel[:,0]*np.cos(theta) + Vel[:,1]*np.sin(theta)
    MeanLong = Vel[:,2]*np.cos(theta) + Vel[:,3]*np.sin(theta)
    TI = np.sqrt((InstLong-MeanLong)**2)/MeanLong

    return TI

  # calculate the Turbulent intensity based on RS or HT
  def calcTI_RS_HT_dir(self,u,v,uInst,vInst,x,y,z,line,point):
    interp = Interp();
    Vel = np.zeros((line.shape[0],4))
    Vel[:,0] = interp.trilinearInterpolation(uInst,x,y,z,line)
    Vel[:,1] = interp.trilinearInterpolation(vInst,x,y,z,line)
    Vel[:,2] = interp.trilinearInterpolation(u,x,y,z,line)
    Vel[:,3] = interp.trilinearInterpolation(v,x,y,z,line)

    HT_UM = interp.trilinearInterpolation(u,x,y,z,point)
    HT_VM = interp.trilinearInterpolation(v,x,y,z,point)

    theta = np.arctan(HT_VM/HT_UM)

    InstLong = Vel[:,0]*np.cos(theta) + Vel[:,1]*np.sin(theta)
    MeanLong = Vel[:,2]*np.cos(theta) + Vel[:,3]*np.sin(theta)
    TI = np.sqrt((InstLong-MeanLong)**2)/MeanLong

    return TI

  # calculate the magnitude of the mean velocity field
  def calcTheta(self,u,v,w,nx,ny,nz):
    theta = np.zeros((nx,ny,nz))

    for i in range(nx-1):
      for j in range(ny-1):
        for k in range(nz-1):
          theta[i,j,k] = np.arctan(v[i,j,k]/u[i,j,k])
    return theta

  def adjustLines(self,line,offset):
    tmp = line
    tmp[:][2] = tmp[:][2]+offset
    return tmp

  # finding the absolute distance to HT and CP
  def findAbsDistX(self,line,ref):
    tmp1 = line[:][0]-ref[0] #x-axis
    tmp2 = line[:][1]-ref[1] #y-axis
    neg_idx = np.where(tmp1<0)[0].max() #find last index that is a negative element
    tmpline = np.sqrt(tmp1**2+tmp2**2)
    tmpline[0:neg_idx] = tmpline[0:neg_idx] * -1
    return tmpline

  # finding the absolute distance to HT and CP
  def findAbsDistXY(self,line,ref):
    tmp1 = line[:][0]-ref[0] #x-axis
    tmp2 = line[:][1]-ref[1] #y-axis
    neg_idx1 = np.where(tmp1<0)[0].max() #find last index that is a negative element
    neg_idx2 = np.where(tmp2<0)[0].max()
    if neg_idx1 > neg_idx2:
      neg_idx = neg_idx1
    else:
      neg_idx = neg_idx2

    tmpline = np.sqrt(tmp1**2+tmp2**2)
    tmpline[0:neg_idx] = tmpline[0:neg_idx] * -1
    return tmpline

  # finding the absolute distance to HT and CP
  def findAbsDist(self,line1):
    line = np.array(line1)
    idx = np.where(line[:,2]==line[:,2].max())[0][0]
    refPt = line[idx,:]
    tmp = line
    tmp[:,0] = tmp[:,0]-refPt[0]
    tmp[:,1] = tmp[:,1]-refPt[1]
    neg_idx1 = np.where(tmp[:,0]<0)[0].max()
    neg_idx2 = np.where(tmp[:,1]<0)[0].max()
    if neg_idx1 > neg_idx2:
      neg_idx = neg_idx1
    else:
      neg_idx = neg_idx2
    tmpline = np.sqrt(tmp[:,0]**2+tmp[:,1]**2)
    tmpline[0:neg_idx] = tmpline[0:neg_idx] * -1
    return tmpline

  def createVerticalLine(self,x_val,y_val,z0,Z):
    index = next(x for x, val in enumerate(Z) if val > z0)
    array = np.ones((len(Z[index:-1]),2))
    array[:,0] *= x_val; array[:,1] *= y_val
    tmp = Z[index:-1].reshape(array.shape[0],1)
    array = np.concatenate((array,tmp),1)
    return array

  # removing the last two elemtns of the passed in array and normalizing z by z0
  def removeLastElement(self,profile,line,z0):
    profile = profile[0:-2] 
    line[:,2] = line[:,2]-z0
    line = line[:-2,:]
    return profile, line

  # cell centering the face center data
  def cellcenter(self,u,v,w,X,Y,Z,dx,dy,dz):
    # averaging calculating
    uCenter = 0.5*(u[1:,:,:]+u[:-1,:,:])
    vCenter = 0.5*(v[:,1:,:]+v[:,:-1,:])
    wCenter = 0.5*(w[:,:,1:]+w[:,:,:-1])
    # adjusting so all velocity components aligne
    uCenter = uCenter[:,1:,1:]
    vCenter = vCenter[1:,:,1:]
    wCenter = wCenter[1:,1:,:]
    # shifting X,Y,Z so that it matched velocities
    X = X[:-1]+0.5*dx
    Y = Y[:-1]+0.5*dy
    Z = Z[:-1]+0.5*dz
    return uCenter, vCenter, wCenter, X, Y, Z

  # cell centering velocities (Part 1)
  def cellcenterVel(self,u,v,w):
    # averaging calculating
    uCenter = 0.5*(u[1:,:,:]+u[:-1,:,:])
    vCenter = 0.5*(v[:,1:,:]+v[:,:-1,:])
    wCenter = 0.5*(w[:,:,1:]+w[:,:,:-1])
    # adjusting so all velocity components aligne
    uCenter = uCenter[:,1:,1:]
    vCenter = vCenter[1:,:,1:]
    wCenter = wCenter[1:,1:,:]
    return uCenter, vCenter, wCenter

  # cell centering the face center data
  def cellcenterCartesian(self,X,Y,Z,dx,dy,dz):
    # shifting X,Y,Z so that it matched velocities
    X = X[:-1]+0.5*dx
    Y = Y[:-1]+0.5*dy
    Z = Z[:-1]+0.5*dz
    return X, Y, Z

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
  def plotContourf(self,data,nx,ny,nz,title,xtitle,ytitle):
    utils = Utils(); settings = Settings('namelist.json')

    fig = plt.figure(); ax = plt.gca()
    t = ax.contourf(data.T)
    fig.colorbar(t)
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)

    fig.savefig(settings.figurePath+title+'.png',dpi=600)

  # lineplots
  def plotFigure(self,x,y,title,xtitle,ytitle,xlim,ylim,fdataname,edataname,filename):
    utils = Utils(); settings = Settings('namelist.json')

    field = utils.readField()
    dataField = field[fdataname]; errUp = field[edataname]
    dataX, dataU, err = utils.creatingErrorBarData(dataField,errUp)

    fig = plt.figure(figsize=(12, 6)); ax = plt.gca()
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'sans-serif'

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
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'sans-serif'

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
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'sans-serif'

    ax.plot(x,y,"r-",label="GIN3D",linewidth=2)
    plt.errorbar(DS,z,xerr=err,fmt='o',label="Field")
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)
    ax.legend(loc="upper right")
    plt.xlim(0.0,1.6); plt.ylim(0,100)
    plt.xticks(np.arange(0,1.6+0.2,0.2)); plt.yticks(np.arange(0,100+20,20))

    fig.savefig(settings.figurePath+filename,dpi=1200)

  def plotLine(self,x,y,title,xtitle,ytitle):
    fig = plt.figure(); ax = plt.gca()
    ax.plot(x,y,"r-",label="GIN3D",linewidth=2)
    ax.set_xlabel(xtitle); ax.set_ylabel(ytitle); ax.set_title(title)
    plt.ylim(y[0],300)

  def plotAlongDirection(self,x,u,v,w,nx,ny,nz):
    fig = plt.figure(); ax = plt.gca()
    ax.plot(x[:-1],u[:-1,np.round(ny/2),np.round(nz/2)],"g-",label="U",linewidth=2)
    ax.plot(x[:-1],v[:-1,np.round(ny/2),np.round(nz/2)],"k-",label="V",linewidth=2)
    ax.plot(x[:-1],w[:-1,np.round(ny/2),np.round(nz/2)],"r-",label="W",linewidth=2)
    ax.legend(loc="upper right")

################################# COMBINED PLOTS #################################
  # lineplots
  def plotFigureBoth(self,S_x,S_y,A_x,A_y,xtitle,ytitle,xlim,ylim,fdataname,edataname,filename):
    utils = Utils(); settings = Settings('namelist.json')

    field = utils.readField()
    dataField = field[fdataname]; errUp = field[edataname]
    dataX, dataU, err = utils.creatingErrorBarData(dataField,errUp)

    fig = plt.figure(figsize=(12, 8)); ax = plt.gca()
    ft = 30
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = True

    plt.errorbar(dataX,(dataU*8.9-8.9)/8.9,yerr=err,fmt='o',label="Field Data")
    ax.plot(S_x,S_y,"g-.",label="Streamwise",linewidth=3)
    ax.plot(A_x,A_y,"r--",label="Angled",linewidth=3)
    ax.set_xlabel(xtitle,fontsize=ft); ax.set_ylabel(ytitle,fontsize=ft)
    plt.xlim(xlim[0],xlim[1]); plt.ylim(ylim[0],ylim[1])
    plt.xticks(fontsize=ft-2); plt.yticks(fontsize=ft-2)
    ax.legend(loc="upper right",fontsize='xx-large')

    fig.savefig(settings.figurePath+filename,dpi=1200)

  # RS line
  def plotRSBoth(self,S_x,S_y,A_x,A_y,xtitle,ytitle,filename):
    utils = Utils(); settings = Settings('namelist.json')

    yloglaw = np.linspace(0,1000,1000)
    roughloglaw = 0.654/0.41 * np.log(yloglaw/0.03) # Log Law used in Figure 7 (DeLeon 2018)

    field = utils.readField()
    kite = field["RSKite"]; cup = field["RSCup"]; gill = field["RSGill"]

    fig = plt.figure(figsize=(10,9)); ax = plt.gca()
    ft = 30
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = True

    ax.semilogy(roughloglaw, yloglaw,"k-", label="LogLaw",linewidth=2)
    ax.semilogy(kite[:,1],kite[:,0],'x',color='b',label="Kite",markersize=10)
    ax.semilogy(cup[:,1],cup[:,0],'.',color='b',label="Cup",markersize=12)
    ax.semilogy(gill[:,1],gill[:,0],'^',color='b',label="Gill",markersize=10)
    ax.semilogy(S_x,S_y,"g-.",label="Streamwise",linewidth=3)
    ax.semilogy(A_x,A_y,"r--",label="Angled",linewidth=3)
    ax.set_xlabel(xtitle,fontsize=ft); ax.set_ylabel(ytitle,fontsize=ft+6)
    ax.legend(loc="upper left",fontsize='xx-large')
    plt.xlim(0.0,20.0); plt.ylim(1e0,1e3)
    plt.xticks(np.arange(0,20+5,5))
    plt.xticks(fontsize=ft-2); plt.yticks(fontsize=ft-2)

    # inset plot
    ax2 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax,[0.6,0.075,0.35,0.35])
    ax2.set_axes_locator(ip)
    ax2.semilogy(roughloglaw, yloglaw,"k-", label="LogLaw",linewidth=2)
    ax2.semilogy(kite[:,1],kite[:,0],'x',color='b',label="Kite",markersize=10)
    ax2.semilogy(cup[:,1],cup[:,0],'.',color='b',label="Cup",markersize=12)
    ax2.semilogy(gill[:,1],gill[:,0],'^',color='b',label="Gill",markersize=10)
    ax2.semilogy(S_x,S_y,"g-.",label="GIN3D",linewidth=3)
    ax2.semilogy(A_x,A_y,"r--",label="GIN3D",linewidth=3)
    ax2.set_xlim(7.0,12.0); ax2.set_ylim(3*1e0,5*1e1)

    fig.savefig(settings.figurePath+filename,dpi=1200)

  # HT line
  def plotHTBoth(self,S_x,S_y,A_x,A_y,xtitle,ytitle,filename):
    utils = Utils(); settings = Settings('namelist.json')

    field = utils.readField()
    dataField = field["HTResults"]
    DS, z, err = utils.errorPropagationCalc(dataField)

    fig = plt.figure(figsize=(10,9)); ax = plt.gca()
    ft = 30
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = True

    plt.errorbar(DS,z,xerr=err,fmt='o',label="Field Data")
    ax.plot(S_x,S_y,"g-.",label="Streamwise",linewidth=3)
    ax.plot(A_x,A_y,"r--",label="Angled",linewidth=3)
    ax.set_xlabel(xtitle,fontsize=ft); ax.set_ylabel(ytitle,fontsize=ft)
    ax.legend(loc="upper right",fontsize='xx-large')
    plt.xlim(0.0,1.6); plt.ylim(0,100)
    plt.xticks(np.arange(0,1.6+0.2,0.2)); plt.yticks(np.arange(0,100+20,20))
    plt.xticks(fontsize=ft-2); plt.yticks(fontsize=ft-2)

    fig.savefig(settings.figurePath+filename,dpi=1200)

  def plotRatioBoth(self,D_x,D_y1,D_y2,D_y3,N_x,N_y1,N_y2,N_y3,xtitle,ytitle,xlim,ylim,filename):
    utils = Utils(); settings = Settings('namelist.json')

#    fig = plt.figure(figsize=(12, 8)); ax = plt.gca()
    fig, ax = plt.subplots(3, sharex=True, sharey=True)
    ft = 30
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['text.usetex'] = True

    norm = np.pi/3

    ax[0].plot(N_x,(N_y3-norm)/norm *100,"r--",linewidth=3)
    ax[0].plot(D_x,(D_y3-norm)/norm *100,"g-.",linewidth=3)
    ax[0].hlines(0.0,-2000,2000,color='k',linestyles='dashed')

    ax[1].plot(N_x,(N_y2-norm)/norm *100,"r--",linewidth=3)
    ax[1].plot(D_x,(D_y2-norm)/norm *100,"g-.",linewidth=3)
    ax[1].hlines(0.0,-2000,2000,color='k',linestyles='dashed')

    ax[2].plot(N_x,(N_y1-norm)/norm *100,"r--",linewidth=3)
    ax[2].plot(D_x,(D_y1-norm)/norm *100,"g-.",linewidth=3)
    ax[2].hlines(0.0,-2000,2000,color='k',linestyles='dashed')

    ax[1].set_ylabel(ytitle,fontsize=ft)
    ax[2].set_xlabel(xtitle,fontsize=ft)

    for i in range(2):
      ax[i].set_xlim(xlim[0],xlim[1])
      ax[i].set_ylim(ylim[0],ylim[1])
#    plt.legend(loc="center left",bbox_to_anchor=(1.04,0.5),borderaxespad=0,fontsize='xx-large')

    neumann = mpatches.Patch(color='red', label='Neumann')
    dirichlet = mpatches.Patch(color='green', label='Dirichlet')
#    plt.legend(handles=[neumann,dirichlet],loc="center left",bbox_to_anchor=(1.04,0.5),borderaxespad=0,fontsize='xx-large')
    plt.legend(handles=[neumann,dirichlet],loc="center",bbox_to_anchor=(0.5, 3.45),ncol=2,fontsize='xx-large')

#    plt.subplots_adjust(top=0.1)
    ax[0].annotate("(a)",xy=(-950,-4),weight='bold',fontsize=16)
    ax[1].annotate("(b)",xy=(-950,-4),weight='bold',fontsize=16)
    ax[2].annotate("(c)",xy=(-950,-4),weight='bold',fontsize=16)

    fig.savefig(settings.figurePath+filename,dpi=1200)

  # Plotting turbulence intensity
  def plotTI(self,ti,S_x,S_y,A_x,A_y,xtitle,ytitle,filename):
    utils = Utils(); settings = Settings('namelist.json')

    if ti == 0: #RS
      levels = np.array([3, 5, 8, 15, 24, 34, 49])
      U = np.array([7.10, 7.86, 8.44, 9.35, 10.19, 10.84, 11.96])
      sigma = np.array([1.42, 1.37, 1.42, 1.27, 1.23, 1.16, 1.16])
    elif ti == 1: #HT
      levels = np.array([1, 3, 5, 8, 15, 24, 34])
      U = np.array([12.99, 15.71, 16.38, 16.30, 16.63, 16.15, 15.77])
      sigma = np.array([1.57, 1.45, 1.38, 1.17, 1.04, 1.19, 1.17])

    TIdata = sigma/U

    fig = plt.figure(figsize=(10,9)); ax = plt.gca()
    ft = 30
#    plt.rcParams['font.family'] = ['serif']
#    plt.rcParams['font.serif'] = ['Times New Roman']
#    plt.rcParams['text.usetex'] = True

    ax.semilogy(TIdata,levels,"x",color='k',label="Field",markersize=10)
    ax.semilogy(S_x[1:],S_y[1:],"g*",label="Streamwise",linewidth=3)
    ax.semilogy(A_x[1:],A_y[1:],"ro",label="Angled",linewidth=3)
    ax.set_xlabel(xtitle,fontsize=ft); ax.set_ylabel(ytitle,fontsize=ft+6)
    ax.legend(loc="upper right",fontsize='xx-large')
    plt.xlim(0.0,0.3); plt.ylim(1e0,1e3)
    plt.xticks(fontsize=ft-2); plt.yticks(fontsize=ft-2)

    fig.savefig(settings.figurePath+filename,dpi=1200)
