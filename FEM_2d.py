#===========================
# 2DFEM Stress Analysis
#===========================
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import sys
import time

input_file = 'Output/2_Delaunay/delaunay2/delaunay2.txt'
output_file = 'Output/3_FEM_2d/fem_2d.txt'

def inpdata_pl3(fnameR,nod,nfree):
    f=open(fnameR,'r')
    text=f.readline()
    text=text.strip()
    text=text.split()
    npoin =int(text[0]) # Number of nodes
    nele  =int(text[1]) # Number of elements
    nsec  =int(text[2]) # Number of sections
    npfix =int(text[3]) # Number of restricted nodes
    nlod  =int(text[4]) # Number of loaded nodes
    nstr  =int(text[5]) # 0: plane strain, 1: plane stress
    # array declaration
    ae    =np.zeros((7,nsec),dtype=float)      # Section characteristics
    node  =np.zeros((nod+1,nele),dtype=int)      # Node-element relationship
    x     =np.zeros((2,npoin),dtype=float)     # Coordinates of nodes
    deltaT=np.zeros(npoin,dtype=float)         # Temperature change of node
    mpfix =np.zeros((nfree,npoin),dtype=int)     # Ristrict conditions
    rdis  =np.zeros((nfree,npoin),dtype=float) # Ristricted displacement
    fp    =np.zeros(nfree*npoin,dtype=float)   # External force vector
    # section characteristics
    for i in range(0,nsec):
        text=f.readline()
        text=text.strip()
        text=text.split()
        ae[0,i]=float(text[0]) #t    : :Plate thickness
        ae[1,i]=float(text[1]) #E    : Elastic modulus
        ae[2,i]=float(text[2]) #po   : Poisson's ratio
        ae[3,i]=float(text[3]) #alpha: Thermal expansion coefficient
        ae[4,i]=float(text[4]) #gamma: Unit weight of material
        ae[5,i]=float(text[5]) #gkh  : Acceleration in x-direction
        ae[6,i]=float(text[6]) #gkv  : Acceleration in y-direction
    # element-node
    for i in range(0,nele):
        text=f.readline()
        text=text.strip()
        text=text.split()
        node[0,i]=int(text[0]) #node_1
        node[1,i]=int(text[1]) #node_2
        node[2,i]=int(text[2]) #node_3
        node[3,i]=int(text[3]) #section characteristic number
    # node coordinates
    for i in range(0,npoin):
        text=f.readline()
        text=text.strip()
        text=text.split()
        x[0,i]   =float(text[0]) # x-coordinate
        x[1,i]   =float(text[1]) # y-coordinate
        deltaT[i]=float(text[2]) # Temperature change of node
    # boundary conditions (0:free, 1:restricted)
    for i in range(0,npfix):
        text=f.readline()
        text=text.strip()
        text=text.split()
        lp=int(text[0])             #fixed node
        mpfix[0,lp-1]=int(text[1])  #fixed in x-direction
        mpfix[1,lp-1]=int(text[2])  #fixed in y-direction
        rdis[0,lp-1]=float(text[3]) #fixed displacement in x-direction
        rdis[1,lp-1]=float(text[4]) #fixed displacement in y-direction
    # load
    if 0<nlod:
        for i in range(0,nlod):
            text=f.readline()
            text=text.strip()
            text=text.split()
            lp=int(text[0])           #loaded node
            fp[2*lp-2]=float(text[1]) #load in x-direction
            fp[2*lp-1]=float(text[2]) #load in y-direction
    f.close()
    return npoin,nele,nsec,npfix,nlod,nstr,ae,node,x,deltaT,mpfix,rdis,fp

def prinp_pl3(fnameW,npoin,nele,nsec,npfix,nlod,nstr,ae,node,x,deltaT,mpfix,rdis,fp):
    fout=open(fnameW,'w')
    # print out of input data
    print('{0:>5s} {1:>5s} {2:>5s} {3:>5s} {4:>5s} {5:>6s}'
    .format('npoin','nele','nsec','npfix','nlod','nstr'),file=fout)
    print('{0:5d} {1:5d} {2:5d} {3:5d} {4:5d} {5:6d}'
    .format(npoin,nele,nsec,npfix,nlod,nstr),file=fout)
    print('{0:>5s} {1:>15s} {2:>15s} {3:>15s} {4:>15s} {5:>15s} {6:>10s} {7:>10s}'
    .format('sec','t','E','po','alpha','gamma','gkh','gkv'),file=fout)
    for i in range(0,nsec):
        print('{0:5d} {1:15.7e} {2:15.7e} {3:15.7e} {4:15.7e} {5:15.7e} {6:10.3f} {7:10.3f}'
        .format(i+1,ae[0,i],ae[1,i],ae[2,i],ae[3,i],ae[4,i],ae[5,i],ae[6,i]),file=fout)
    print('{0:>5s} {1:>15s} {2:>15s} {3:>15s} {4:>15s} {5:>15s} {6:>5s} {7:>5s}'
    .format('node','x','y','fx','fy','deltaT','kox','koy'),file=fout)
    for i in range(0,npoin):
        lp=i+1
        print('{0:5d} {1:15.7e} {2:15.7e} {3:15.7e} {4:15.7e} {5:15.7e} {6:5d} {7:5d}'
        .format(lp,x[0,i],x[1,i],fp[2*i],fp[2*i+1],deltaT[i],mpfix[0,i],mpfix[1,i]),file=fout)
    print('{0:>5s} {1:>5s} {2:>5s} {3:>15s} {4:>15s}'
    .format('node','kox','koy','rdis_x','rdis_y'),file=fout)
    for i in range(0,npoin):
        if 0<mpfix[0,i]+mpfix[1,i]:
            lp=i+1
            print('{0:5d} {1:5d} {2:5d} {3:15.7e} {4:15.7e}'
            .format(lp,mpfix[0,i],mpfix[1,i],rdis[0,i],rdis[1,i]),file=fout)
    print('{0:>5s} {1:>5s} {2:>5s} {3:>5s} {4:>5s} {5:>5s}'
    .format('elem','i','j','k','l','sec'),file=fout)
    for ne in range(0,nele):
        print('{0:5d} {1:5d} {2:5d} {3:5d} {4:5d}'
        .format(ne+1,node[0,ne],node[1,ne],node[2,ne],node[3,ne]),file=fout)
    fout.close()


def prout_pl3(fnameW,npoin,nele,disg,strs):
    fout=open(fnameW,'a')
    # displacement
    print('{0:>5s} {1:>15s} {2:>15s}'.format('node','dis-x','dis-y'),file=fout)
    for i in range(0,npoin):
        lp=i+1
        print('{0:5d} {1:15.7e} {2:15.7e}'.format(lp,disg[2*lp-2],disg[2*lp-1]),file=fout)
    # section force
    print('{0:>5s} {1:>15s} {2:>15s} {3:>15s} {4:>15s} {5:>15s} {6:>15s}'
    .format('elem','sig_x','sig_y','tau_xy','p1','p2','ang'),file=fout)
    for ne in range(0,nele):
        sigx =strs[0,ne]
        sigy =strs[1,ne]
        tauxy=strs[2,ne]
        ps1,ps2,ang=pst_pl3(sigx,sigy,tauxy)
        print('{0:5d} {1:15.7e} {2:15.7e} {3:15.7e} {4:15.7e} {5:15.7e} {6:15.7e}'
        .format(ne+1,sigx,sigy,tauxy,ps1,ps2,ang),file=fout)
    fout.close()


def dmat_pl3(nstr,E,po):
    d=np.zeros((3,3),dtype=float)
    if nstr==0: #plane strain
        d[0,0]=1-po; d[0,1]=po
        d[1,0]=po  ; d[1,1]=1-po
        d[2,2]=0.5*(1-2*po)
        d=E/(1+po)/(1-2*po)*d
    else: # plane stress
        d[0,0]=1  ; d[0,1]=po
        d[1,0]=po ; d[1,1]=1
        d[2,2]=0.5*(1-po)
        d=E/(1-po**2)*d
    return d




def bmat_pl3(x1,y1,x2,y2,x3,y3):
    bm=np.zeros((3,6),dtype=float)
    area=0.5*((x3-x2)*y1+(x1-x3)*y2+(x2-x1)*y3)
    # if -0.00001 < area and area < 0.00001:
    #     print('area:   ',area)
    #     print('x1:   ', x1)
    #     print('x2:   ', x2)
    #     print('x3:   ', x3)
    #     print('y1:   ', y1)
    #     print('y2:   ', y2)
    #     print('y3:   ', y3)

    bm[0,0]=y2-y3; bm[0,1]=0    ; bm[0,2]=y3-y1; bm[0,3]=0    ; bm[0,4]=y1-y2; bm[0,5]=0
    bm[1,0]=0    ; bm[1,1]=x3-x2; bm[1,2]=0    ; bm[1,3]=x1-x3; bm[1,4]=0    ; bm[1,5]=x2-x1
    bm[2,0]=x3-x2; bm[2,1]=y2-y3; bm[2,2]=x1-x3; bm[2,3]=y3-y1; bm[2,4]=x2-x1; bm[2,5]=y1-y2
    bm=bm/2/area
    return bm,area


def sm_pl3(nstr,t,E,po,x1,y1,x2,y2,x3,y3):
    #Stiffness matrix [sm]=[B]T[D][B]*t*det(J)
    d=dmat_pl3(nstr,E,po)
    bm,area=bmat_pl3(x1,y1,x2,y2,x3,y3)
    sm=np.dot(bm.T,np.dot(d,bm))*t*area
    return sm


def calst_pl3(nstr,E,po,alpha,tem,wd,x1,y1,x2,y2,x3,y3):
    eps0=np.zeros(3,dtype=np.float64)
    #stress vector {stress}=[D][B]{u}
    d=dmat_pl3(nstr,E,po)
    bm,area=bmat_pl3(x1,y1,x2,y2,x3,y3)
    eps=np.dot(bm,wd)
    #Thermal strain
    if nstr==0: # plane strain
        eps0[0]=tem*(1.0+po)*alpha
        eps0[1]=eps0[0]
        eps0[2]=0.0
    else: # plane stress
        eps0[0]=tem*alpha
        eps0[1]=eps0[0]
        eps0[2]=0.0
    stress=np.dot(d,(eps-eps0))
    return stress


def pst_pl3(sigx,sigy,tauxy):
    ps1=0.5*(sigx+sigy)+np.sqrt(0.25*(sigx-sigy)*(sigx-sigy)+tauxy*tauxy)
    ps2=0.5*(sigx+sigy)-np.sqrt(0.25*(sigx-sigy)*(sigx-sigy)+tauxy*tauxy)
    if sigx==sigy:
        if tauxy >0.0: ang= 45.0
        if tauxy <0.0: ang=135.0
        if tauxy==0.0: ang=  0.0
    else:
        ang=0.5*np.arctan(2.0*tauxy/(sigx-sigy))
        ang=180.0/np.pi*ang
        if sigx>sigy and tauxy>=0.0: ang=ang
        if sigx>sigy and tauxy <0.0: ang=ang+180.0
        if sigx<sigy:                ang=ang+90.0
    return ps1,ps2,ang


def tfvec_pl3(nstr,t,E,po,alpha,tem,x1,y1,x2,y2,x3,y3):
    eps0=np.zeros(3,dtype=np.float64)
    # {tfe=[B]T[D]{eps0}
    d=dmat_pl3(nstr,E,po)
    bm,area=bmat_pl3(x1,y1,x2,y2,x3,y3)
    #Thermal strain
    if nstr==0: # plane strain
        eps0[0]=tem*(1.0+po)*alpha
        eps0[1]=eps0[0]
        eps0[2]=0.0
    else: # plane stress
        eps0[0]=tem*alpha
        eps0[1]=eps0[0]
        eps0[2]=0.0
    tfe=np.dot(bm.T,np.dot(d,eps0))*t*area
    return tfe


def bfvec_pl3(t,gamma,gkh,gkv,x1,y1,x2,y2,x3,y3):
    mat=np.zeros((6,6),dtype=np.float64)
    area=0.5*((x3-x2)*y1+(x1-x3)*y2+(x2-x1)*y3)
    mat[0,0]=0.50;mat[0,1]=0.00;mat[0,2]=0.25;mat[0,3]=0.00;mat[0,4]=0.25;mat[0,5]=0.00
    mat[1,0]=0.00;mat[1,1]=0.50;mat[1,2]=0.00;mat[1,3]=0.25;mat[1,4]=0.00;mat[1,5]=0.25
    mat[2,0]=0.25;mat[2,1]=0.00;mat[2,2]=0.50;mat[2,3]=0.00;mat[2,4]=0.25;mat[2,5]=0.00
    mat[3,0]=0.00;mat[3,1]=0.25;mat[3,2]=0.00;mat[3,3]=0.50;mat[3,4]=0.00;mat[3,5]=0.25
    mat[4,0]=0.25;mat[4,1]=0.00;mat[4,2]=0.25;mat[4,3]=0.00;mat[4,4]=0.50;mat[4,5]=0.00
    mat[5,0]=0.00;mat[5,1]=0.25;mat[5,2]=0.00;mat[5,3]=0.25;mat[5,4]=0.00;mat[5,5]=0.50
    mat=mat*gamma*t*area/3
    w=np.array([gkh,gkv,gkh,gkv,gkh,gkv],dtype=np.float64)
    bfe=np.dot(mat,w)
    return bfe


def main_pl3():
    start=time.time()
    fnameR = input_file
    fnameW = output_file
    nod=3          # Number of nodes per element
    nfree=2        # Degree of freedom per node
    # data input
    npoin,nele,nsec,npfix,nlod,nstr,ae,node,x,deltaT,mpfix,rdis,fp=inpdata_pl3(fnameR,nod,nfree)
    print(x)
    # print out of input data
    prinp_pl3(fnameW,npoin,nele,nsec,npfix,nlod,nstr,ae,node,x,deltaT,mpfix,rdis,fp)
    # array declaration
    ir=np.zeros(nod*nfree,dtype=int)         # Work vector for matrix assembly
    gk=np.zeros((nfree*npoin,nfree*npoin),dtype=float)   # Global stiffness matrix
    # assembly of stiffness matrix & load vector
    for ne in range(0,nele):
        i=node[0,ne]-1
        j=node[1,ne]-1
        k=node[2,ne]-1
        m=node[3,ne]-1
        x1=x[0,i]; y1=x[1,i]
        x2=x[0,j]; y2=x[1,j]
        x3=x[0,k]; y3=x[1,k]
        t    =ae[0,m]
        E    =ae[1,m]
        po   =ae[2,m]
        alpha=ae[3,m]
        gamma=ae[4,m]
        gkh  =ae[5,m]
        gkv  =ae[6,m]
        tem=(deltaT[i]+deltaT[j]+deltaT[k])/3   # average temperature change
        sm=sm_pl3(nstr,t,E,po,x1,y1,x2,y2,x3,y3)  # element stiffness matrix
        tfe=tfvec_pl3(nstr,t,E,po,alpha,tem,x1,y1,x2,y2,x3,y3) # thermal load vector
        bfe=bfvec_pl3(t,gamma,gkh,gkv,x1,y1,x2,y2,x3,y3) # body force vector
        ir[5]=2*k+1; ir[4]=ir[5]-1
        ir[3]=2*j+1; ir[2]=ir[3]-1
        ir[1]=2*i+1; ir[0]=ir[1]-1
        for i in range(0,nod*nfree):
            it=ir[i]
            fp[it]=fp[it]+tfe[i]+bfe[i]
            for j in range(0,nod*nfree):
                jt=ir[j]
                gk[it,jt]=gk[it,jt]+sm[i,j]
    # treatment of boundary conditions
    for i in range(0,npoin):
        for j in range(0,nfree):
            if mpfix[j,i]==1:
                iz=i*nfree+j
                fp[iz]=0.0
    for i in range(0,npoin):
        for j in range(0,nfree):
            if mpfix[j,i]==1:
                iz=i*nfree+j
                fp=fp-rdis[j,i]*gk[:,iz]
                gk[:,iz]=0.0
                gk[iz,iz]=1.0
    # solution of simultaneous linear equations
    #disg = np.linalg.solve(gk, fp)
    gk = csr_matrix(gk)
    disg = spsolve(gk, fp, use_umfpack=True)
    # recovery of restricted displacements
    for i in range(0,npoin):
        for j in range(0,nfree):
            if mpfix[j,i]==1:
                iz=i*nfree+j
                disg[iz]=rdis[j,i]
    # calculation of element stress
    strs=np.zeros((3,nele),dtype=np.float64) # Section force vector
    wd  =np.zeros(6,dtype=np.float64)        # element displacement
    for ne in range(0,nele):
        i=node[0,ne]-1
        j=node[1,ne]-1
        k=node[2,ne]-1
        m=node[3,ne]-1
        x1=x[0,i]; y1=x[1,i]
        x2=x[0,j]; y2=x[1,j]
        x3=x[0,k]; y3=x[1,k]
        wd[0]=disg[2*i]; wd[1]=disg[2*i+1]
        wd[2]=disg[2*j]; wd[3]=disg[2*j+1]
        wd[4]=disg[2*k]; wd[5]=disg[2*k+1]
        E    =ae[1,m]
        po   =ae[2,m]
        alpha=ae[3,m]
        tem=(deltaT[i]+deltaT[j]+deltaT[k])/3
        strs[:,ne]=calst_pl3(nstr,E,po,alpha,tem,wd,x1,y1,x2,y2,x3,y3)
        print('ne:  ', ne, '  strs:   ', strs[:,ne])
    prout_pl3(fnameW,npoin,nele,disg,strs)
    # information
    dtime=time.time()-start
    print('n={0}  time={1:.3f}'.format(nfree*npoin,dtime)+' sec')
    fout=open(fnameW,'a')
    print('n={0}  time={1:.3f}'.format(nfree*npoin,dtime)+' sec',file=fout)
    fout.close()

#==============
# Execution
#==============
if __name__ == '__main__': 
    main_pl3()
