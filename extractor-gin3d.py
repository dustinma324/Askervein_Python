import numpy as np
import pandas as pd

# prefix of file names
#fileprefix = ["AskTInlet_35m_co_sbc"]
fileprefix = ["Askervein_Stream"]

# desired timesteps. may give "final" as a timestep
timestep = ["0130000"]

# desired quantities to investigate 
quantity = ["U", "V", "W"]
#quantity = ["DF"]
#quantity = ["uu","vv","ww"]

# is the quantity a time averaged field (True of False)
# must be same length as quantity
timeavgflags = [True, True, True]
#timeavgflags = [False]

# file suffix (must be same length as quantity)
filesuffix = [".dat", ".dat", ".dat"]
#filesuffix = [".dat"]

# grid size per given prefix
gridsize = [641, 513, 257]
gridsize = np.array(gridsize).reshape(len(gridsize)/3,3)

# dimensions per given prefix
dimensions = [8128.0, 5800.0, 1000.0]
dimensions = np.array(dimensions).reshape(len(dimensions)/3,3)

# Flag for vertical z profile at a given x and y coordinate
#VPFlag = True
VPFlag = False
VPLocs = [
#        150.7, 2716.8,     # RS for 6500
        1778.88, 2716.8,   # RS for 8128
#        3075.0,3166.0,     # HT for 6500
        3075.0+1628.18,3166.0,  # HT for 8128
#        3075.0+1628.18,1000.0,  # Ref aligned with HT for 8128 
#        3175.0,3096.0,
#         0.0, 2700.0,
        ]
VPLocs = np.array(VPLocs).reshape(len(VPLocs)/2,2)

# Flag for a set of x, y and z coordinates in a CSV
SCFlag = True
#SCFlag = False
#SCFile = "LinesForSampling/RS_site_7800x5800_10m.csv"
#SCFile = "LinesForSampling/Askervein_7800mx5800m_AALine.csv"
#SCFile = "LinesForSampling/Askervein_7800mx5800m_ALine.csv"
#SCFile = "LinesForSampling/Askervein_7800mx5800m_BLine.csv"
#SCFile = "LinesForSampling/Askervein_6500mx5800m_up1.0m_ALine.csv"
#SCFile = "LinesForSampling/Askervein_6500mx5800m_up1.0m_AALine.csv"
#SCFile = "LinesForSampling/RS_site_6500x5800_up1.0m_10m.csv"
#SCFile = "LinesForSampling/Askervein_8128mx5800m_up1.0m_ALine.csv"
#SCFile = "LinesForSampling/Askervein_8128mx5800m_up1.0m_AALine.csv"
#SCFile = "LinesForSampling/RS_site_8128x5800_up1.0m_10m.csv"
SCFile = "ALineProfileRegular.csv"
#SCFile = "LinesForSampling/ProfileRegular_8128mx5800m_up1.0m_AALine.csv"
#SCFile = "LinesForSampling/ProfileRegular_8128mx5800m_up1.0m_ALine.csv"

#Power Interpolation in Vertical
PIFlag = True

if SCFlag:
    Coords = pd.read_csv(SCFile, header=None).as_matrix()
    # Hack to move points up for long domain
    Coords[:,0] = Coords[:,0] + 1628.18
    Coords[:,2] = Coords[:,2] + 1.0
    print Coords

for p in range(len(fileprefix)):
    for r in range(len(timestep)):
        for q in range(len(quantity)):
            fn = fileprefix[p] + "_" + quantity[q]
            if timeavgflags[q]:
                fn += "_mean"
            fn += "_" + timestep[r] + filesuffix[q]

            print fn
            nx = gridsize[p,0]
            ny = gridsize[p,1]
            nz = gridsize[p,2]
            dx = dimensions[p,0] / (nx-2)
            dy = dimensions[p,1] / (ny-2)
            dz = dimensions[p,2] / (nz-2)
            if filesuffix[q] == ".dat":
                A = pd.read_csv(fn, sep = " ", header=None)
                del(A[nx])
                A = A.as_matrix().reshape(nz,ny,nx)
            else:
                A = np.fromfile(fn, dtype="float64", count=nx*ny*nz).reshape(nz,ny,nx)

            # based on an x and y coordinate, find vertical profile with values
            # coincident with the height (z-direction) of a cell center 
            if VPFlag:
                # cycle through the coordinate list 
                for t in range(VPLocs.shape[0]):
                    # find the closest staggered grid indices and the normalized
                    # distance for interpolation
                    if quantity[q] == "U":
                        x1 = np.floor(VPLocs[t,0]/dx)
                        x2 = x1 + 1 
                        ax = VPLocs[t,0]/dx - x1
                    else:
                        x1 = np.floor((VPLocs[t,0]+0.5*dx)/dx)
                        x2 = x1 + 1 
                        ax = VPLocs[t,0]/dx - x1 + 0.5 
                    if quantity[q] == "V":
                        y1 = np.floor(VPLocs[t,1]/dy)
                        y2 = y1 + 1 
                        ay = VPLocs[t,1]/dy - y1
                    else:
                        y1 = np.floor((VPLocs[t,1]+0.5*dy)/dy)
                        y2 = y1 + 1 
                        ay = VPLocs[t,1]/dy - y1 + 0.5
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    print quantity[q],x1,x2,y1,y2,ax,ay
                    # use normalized distances to perform bilinear interpolation 
                    print A[:,y1,x1]
                    ylo = (1.0-ax)*A[:,y1,x1] + ax*A[:,y1,x2]
                    yhi = (1.0-ax)*A[:,y2,x1] + ax*A[:,y2,x2]
                    P = (1.0-ay)*ylo + ay*yhi
                    # average to cell center height (z-direction)
                    if quantity[q] == "W":
                        P = 0.5*(P[1:] + P[:-1])
                    # write the profile 
                    np.savetxt(fn[:-3] + "x" + str(VPLocs[t,0]) + ".y" + str(VPLocs[t,1]) + ".profile",
                            P, fmt="%.16e")

            # Based on a 3D coordinate, use trilinear interpolation to bring values to coordinate 
            if SCFlag:
                # cycle through the coordinate list 
                phi = []
                for t in range(Coords.shape[0]):
                    # heights for logarithmic interpolation 
                    zloglo = 0.0
                    zloghi = 0.0
                    # find the closest staggered grid indices and the normalized
                    # distance for interpolation
                    if quantity[q] == "U":
                        x1 = np.floor(Coords[t,0]/dx)
                        x2 = x1 + 1 
                        ax = Coords[t,0]/dx - x1
                    else:
                        x1 = np.floor((Coords[t,0]+0.5*dx)/dx)
                        x2 = x1 + 1 
                        ax = Coords[t,0]/dx - x1 + 0.5 
                    if quantity[q] == "V":
                        y1 = np.floor(Coords[t,1]/dy)
                        y2 = y1 + 1 
                        ay = Coords[t,1]/dy - y1
                    else:
                        y1 = np.floor((Coords[t,1]+0.5*dy)/dy)
                        y2 = y1 + 1 
                        ay = Coords[t,1]/dy - y1 + 0.5
                    if quantity[q] == "W":
                        z1 = np.floor(Coords[t,2]/dz)
                        z2 = z1 + 1 
                        az = Coords[t,2]/dz - z1
                        zloglo = z1*dz
                        zloghi = z2*dz
                    else:
                        z1 = np.floor((Coords[t,2]+0.5*dz)/dz)
                        z2 = z1 + 1 
                        az = Coords[t,2]/dz - z1 + 0.5
                        zloglo = (z1-0.5)*dz
                        zloghi = (z2-0.5)*dz
                    print quantity[q],x1,x2,y1,y2,z1,z2,ax,ay,az
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    z1 = int(z1)
                    z2 = int(z2)
                    # use normalized distances to perform trilinear interpolation 
                    # bottom, south interpolation in x 
                    bsi = (1.0-ax)*A[z1,y1,x1] + ax*A[z1,y1,x2] 
                    # bottom, north interpolation in x 
                    bni = (1.0-ax)*A[z1,y2,x1] + ax*A[z1,y2,x2] 
                    # top, south interpolation in x 
                    tsi = (1.0-ax)*A[z2,y1,x1] + ax*A[z2,y1,x2] 
                    # top, north interpolation in x 
                    tni = (1.0-ax)*A[z2,y2,x1] + ax*A[z2,y2,x2] 
                    # bottom interpolation in y
                    bj = (1.0-ay)*bsi + ay*bni
                    # top interpolation in y 
                    tj = (1.0-ay)*tsi + ay*tni
                    # interpolation in z 
                    if PIFlag:
                        # Check if we get ratios zero or below. If so, resort to linear instead.
			print bj/tj, zloglo/zloghi 
                        if (bj/tj) > 0.01:
                            pn = np.log10(bj/tj) / np.log10(zloglo/zloghi) 
			    print pn 
                            phi.append(tj * np.power(Coords[t,2]/zloghi,pn) )
                        else:
                            phi.append( (1.0-az)*bj + az*tj )
                    else:
                        phi.append( (1.0-az)*bj + az*tj )

                print "Saving  " + fn[:-4] + "_from_" + SCFile[17:]
                np.savetxt(fn[:-4] + "_from_" + SCFile[17:], phi, fmt="%.16e") 
