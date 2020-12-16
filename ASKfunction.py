import json
import numpy as np

class Settings:
  def __init__(self,namelist):
    with open(namelist) as json_file:
      data = json.load(json_file)
    self.nx = data["nx"]
    self.ny = data["ny"]
    self.nz = data["nz"]
    self.lx = data["lx"]
    self.ly = data["ly"]
    self.lz = data["lz"]
    self.U = data["UPath"]
    self.V = data["VPath"]
    self.W = data["WPath"]

class Utils:

  def read2Dmesh(self,u,v,w,nx,ny,nz):

    for i in range(nx):
      for j in range(ny):
        for k in range(nz):
          mag[] = np.sqrt(u[]**2 + v[]**2 + w[]**2)

    magnitude = {
        'magnitude': mag
    }
     
