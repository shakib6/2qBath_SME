import numpy as np
import math, cmath
from scipy import linalg as LA
from input_data import *


def rho_init(a=0.0,b=0.0,rho_c1=0.5*np.identity(2),rho_c2=0.5*np.identity(2)):
    if a == 0 and b == 0:
        rho_c = np.kron(rho_c1,rho_c2)
    else:
        rho_c[0,0] = (a+b)**2/2.0
        rho_c[3,3] = (a-b)**2/2.0
        rho_c[0,3] = (a**2-b**2)/2.0
        rho_c[3,0] = (a**2-b**2)/2.0
    result = rho_c   
    return result


# -------------------------------------------------------------------------------------
#              Local measurement basis, Eq.(42) August 2018 draft
# -------------------------------------------------------------------------------------
def local_maps(rho_c):
    rho_c_u1 = abs(b_ee)**2*rho_c - 0.5*abs(b_ee)**2*(g1*(rho_c@L1@L1d+L1@L1d@rho_c)\
                                                      +g2*(rho_c@L2@L2d+L2@L2d@rho_c))\
                -math.sqrt(g1*g2)*(b_gg*np.conj(b_ee)*L1@L2@rho_c + b_ee*np.conj(b_gg)*rho_c@L1d@L2d)
    rho_c_u2 = g1*abs(b_gg)**2*L1@rho_c@L1d + g2*abs(b_ee)**2*L2d@rho_c@L2 \
                            + math.sqrt(g1*g2)*(b_gg*np.conj(b_ee)*L1@rho_c@L2 + b_ee*np.conj(b_gg)*L2d@rho_c@L1d)
    rho_c_u3 = g1*abs(b_ee)**2*L1d@rho_c@L1 + g2*abs(b_gg)**2*L2@rho_c@L2d \
                            + math.sqrt(g1*g2)*(b_gg*np.conj(b_ee)*L2@rho_c@L1 + b_ee*np.conj(b_gg)*L1d@rho_c@L2d)
    rho_c_u4 = abs(b_gg)**2*rho_c - 0.5*abs(b_gg)**2*(g1*(rho_c@L1d@L1+L1d@L1@rho_c)\
                                                      +g2*(rho_c@L2d@L2+L2d@L2@rho_c))\
                -math.sqrt(g1*g2)*(b_gg*np.conj(b_ee)*rho_c@L1@L2 + b_ee*np.conj(b_gg)*L1d@L2d@rho_c)
    return (rho_c_u1,rho_c_u2,rho_c_u3,rho_c_u4)


# -------------------------------------------------------------------------------------
#                 Joint measurement basis, Eq.(45) August 2018 draft
# -------------------------------------------------------------------------------------
def joint_maps(rho_c):
    rho_c_u1 = 0.5*((alpha_plus + beta_plus)*rho_c - 0.5*g1*(alpha_plus*rho_c@L1@L1d+beta_plus*rho_c@L1d@L1 + \
                                       np.conj(alpha_plus)*L1@L1d@rho_c+np.conj(beta_plus)*L1d@L1@rho_c)+\
                                              - 0.5*g2*(alpha_plus*rho_c@L2@L2d+beta_plus*rho_c@L2d@L2 + \
                                       np.conj(alpha_plus)*L2@L2d@rho_c+np.conj(beta_plus)*L2d@L2@rho_c)-\
                                    math.sqrt(g1*g2)*(alpha_plus*rho_c@L1@L2+beta_plus*rho_c@L1d@L2d +\
                                        np.conj(alpha_plus)*L2d@L1d@rho_c+np.conj(beta_plus)*L2@L1@rho_c))
    rho_c_u2 = 0.5*((alpha_minus - beta_minus)*rho_c - 0.5*g1*(alpha_minus*rho_c@L1@L1d-beta_minus*rho_c@L1d@L1 + \
                                       np.conj(alpha_minus)*L1@L1d@rho_c-np.conj(beta_minus)*L1d@L1@rho_c)+\
                                                - 0.5*g2*(alpha_minus*rho_c@L2@L2d-beta_minus*rho_c@L2d@L2 + \
                                       np.conj(alpha_minus)*L2@L2d@rho_c-np.conj(beta_minus)*L2d@L2@rho_c)+\
                                    math.sqrt(g1*g2)*(alpha_minus*rho_c@L1@L2-beta_minus*rho_c@L1d@L2d +\
                                        np.conj(alpha_minus)*L2d@L1d@rho_c-np.conj(beta_minus)*L2@L1@rho_c))
    rho_c_u3 = 0.5*(g1*(b_gg*L1+b_ee*L1d)@rho_c@(np.conj(b_gg)*L1d+np.conj(b_ee)*L1)+\
                    g2*(b_gg*L2+b_ee*L2d)@rho_c@(np.conj(b_gg)*L2d+np.conj(b_ee)*L2) + \
                math.sqrt(g1*g2)*((b_gg*L1+b_ee*L1d)@rho_c@(np.conj(b_gg)*L2d+np.conj(b_ee)*L2) +\
                                  (b_gg*L2+b_ee*L2d)@rho_c@(np.conj(b_gg)*L1d+np.conj(b_ee)*L1)))
    rho_c_u4 = 0.5*(g1*(b_gg*L1-b_ee*L1d)@rho_c@(np.conj(b_gg)*L1d-np.conj(b_ee)*L1) + \
                    g2*(b_gg*L2-b_ee*L2d)@rho_c@(np.conj(b_gg)*L2d-np.conj(b_ee)*L2) - \
                math.sqrt(g1*g2)*((b_gg*L1-b_ee*L1d)@rho_c@(np.conj(b_gg)*L2d-np.conj(b_ee)*L2) +\
                                  (b_gg*L2-b_ee*L2d)@rho_c@(np.conj(b_gg)*L1d-np.conj(b_ee)*L1)))
    return (rho_c_u1,rho_c_u2,rho_c_u3,rho_c_u4)
# def joint_maps(rho_c):
#     rho_c_u1 = Kraus_opt(name)[0]@rho_c@np.transpose(np.conj(Kraus_opt(name)[0]))
#     rho_c_u2 = Kraus_opt(name)[1]@rho_c@np.transpose(np.conj(Kraus_opt(name)[1]))
#     rho_c_u3 = Kraus_opt(name)[2]@rho_c@np.transpose(np.conj(Kraus_opt(name)[2]))
#     rho_c_u4 = Kraus_opt(name)[3]@rho_c@np.transpose(np.conj(Kraus_opt(name)[3]))

#     return (rho_c_u1,rho_c_u2,rho_c_u3,rho_c_u4)
# -------------------------------------------------------------------------------------
#                                   Energy-subspace basis
# -------------------------------------------------------------------------------------
def energy_subspace_maps(rho_c):
    rho_c_u1 = 0.5*((alpha_plus + beta_plus)*rho_c - 0.5*g1*(alpha_plus*rho_c@L1@L1d+beta_plus*rho_c@L1d@L1 + \
                                       np.conj(alpha_plus)*L1@L1d@rho_c+np.conj(beta_plus)*L1d@L1@rho_c)+\
                                              - 0.5*g2*(alpha_plus*rho_c@L2@L2d+beta_plus*rho_c@L2d@L2 + \
                                       np.conj(alpha_plus)*L2@L2d@rho_c+np.conj(beta_plus)*L2d@L2@rho_c)-\
                                    math.sqrt(g1*g2)*(alpha_plus*rho_c@L1@L2+beta_plus*rho_c@L1d@L2d +\
                                        np.conj(alpha_plus)*L2d@L1d@rho_c+np.conj(beta_plus)*L2@L1@rho_c))+\
               0.5*((alpha_minus - beta_minus)*rho_c - 0.5*g1*(alpha_minus*rho_c@L1@L1d-beta_minus*rho_c@L1d@L1 + \
                                       np.conj(alpha_minus)*L1@L1d@rho_c-np.conj(beta_minus)*L1d@L1@rho_c)+\
                                                - 0.5*g2*(alpha_minus*rho_c@L2@L2d-beta_minus*rho_c@L2d@L2 + \
                                       np.conj(alpha_minus)*L2@L2d@rho_c-np.conj(beta_minus)*L2d@L2@rho_c)+\
                                    math.sqrt(g1*g2)*(alpha_minus*rho_c@L1@L2-beta_minus*rho_c@L1d@L2d +\
                                        np.conj(alpha_minus)*L2d@L1d@rho_c-np.conj(beta_minus)*L2@L1@rho_c))
    rho_c_u2 = 0.5*(g1*(b_gg*L1+b_ee*L1d)@rho_c@(np.conj(b_gg)*L1d+np.conj(b_ee)*L1)+\
                    g2*(b_gg*L2+b_ee*L2d)@rho_c@(np.conj(b_gg)*L2d+np.conj(b_ee)*L2) + \
                math.sqrt(g1*g2)*((b_gg*L1+b_ee*L1d)@rho_c@(np.conj(b_gg)*L2d+np.conj(b_ee)*L2) +\
                                  (b_gg*L2+b_ee*L2d)@rho_c@(np.conj(b_gg)*L1d+np.conj(b_ee)*L1)))+\
                0.5*(g1*(b_gg*L1-b_ee*L1d)@rho_c@(np.conj(b_gg)*L1d-np.conj(b_ee)*L1) + \
                    g2*(b_gg*L2-b_ee*L2d)@rho_c@(np.conj(b_gg)*L2d-np.conj(b_ee)*L2) - \
                math.sqrt(g1*g2)*((b_gg*L1-b_ee*L1d)@rho_c@(np.conj(b_gg)*L2d-np.conj(b_ee)*L2) +\
                                  (b_gg*L2-b_ee*L2d)@rho_c@(np.conj(b_gg)*L1d-np.conj(b_ee)*L1)))
    return (rho_c_u1,rho_c_u2,rho_c_u3,rho_c_u4)
            
            
channel = {1:local_maps, 2:joint_maps, 3:energy_subspace_maps}           

# -------------------------------------------------------------------------------------
#                  Kraus operators for local measurement basis
# -------------------------------------------------------------------------------------
def Kraus_opt(name):
    if name == "local":
        K_u1 = b_ee*np.identity(4) - 1.0j*math.sqrt(g1)*b_ge*L1 - 1.0j*math.sqrt(g2)*b_eg*L2\
            - 0.5*g1*b_ee*L1@L1d - 0.5*g2*b_ee*L2@L2d - math.sqrt(g1*g2)*b_gg*L1@L2
        K_u2 = b_eg*np.identity(4) - 1.0j*math.sqrt(g1)*b_gg*L1 - 1.0j*math.sqrt(g2)*b_ee*L2d\
            - 0.5*g1*b_eg*L1@L1d - 0.5*g2*b_eg*L2d@L2 - math.sqrt(g1*g2)*b_ge*L1@L2d
        K_u3 = b_ge*np.identity(4) - 1.0j*math.sqrt(g1)*b_ee*L1d - 1.0j*math.sqrt(g2)*b_gg*L2\
            - 0.5*g1*b_ge*L1d@L1 - 0.5*g2*b_ge*L2d@L2 - math.sqrt(g1*g2)*b_eg*L1d@L2
        K_u4 = b_gg*np.identity(4) - 1.0j*math.sqrt(g1)*b_eg*L1d - 1.0j*math.sqrt(g2)*b_ge*L2d\
            - 0.5*g1*b_gg*L1d@L1 - 0.5*g2*b_gg*L2d@L2 - math.sqrt(g1*g2)*b_ee*L1d@L2d
    elif name == "joint":  
        K_u1 = 1.0/math.sqrt(2.0)*(b_ee+b_gg)*np.identity(4)-g1/(2.0*math.sqrt(2))*(b_ee*L1@L1d+b_gg*L1d@L1)\
                -g2/(2.0*math.sqrt(2))*(b_ee*L2@L2d+b_gg*L2d@L2)-math.sqrt(0.5*g1*g2)*(b_gg*L1@L2+b_ee*L1d@L2d)
        K_u2 = 1.0/math.sqrt(2.0)*(b_ee-b_gg)*np.identity(4)-g1/(2.0*math.sqrt(2))*(b_ee*L1@L1d-b_gg*L1d@L1)\
                -g2/(2.0*math.sqrt(2))*(b_ee*L2@L2d-b_gg*L2d@L2)-math.sqrt(0.5*g1*g2)*(b_gg*L1@L2-b_ee*L1d@L2d)
        K_u3 = -1.0j/math.sqrt(2.0)*(math.sqrt(g1)*(b_gg*L1+b_ee*L1d) + math.sqrt(g2)*(b_gg*L2+b_ee*L2d))
        K_u4 = -1.0j/math.sqrt(2.0)*(math.sqrt(g1)*(b_gg*L1-b_ee*L1d) - math.sqrt(g2)*(b_gg*L2-b_ee*L2d))
    else: # energy-subsapce basis
        K_u1 = 1.0/math.sqrt(2.0)*(b_ee+b_gg)*np.identity(4)-g1/(2.0*math.sqrt(2))*(b_ee*L1@L1d+b_gg*L1d@L1)\
                -g2/(2.0*math.sqrt(2))*(b_ee*L2@L2d+b_gg*L2d@L2)-math.sqrt(0.5*g1*g2)*(b_gg*L1@L2+b_ee*L1d@L2d)
        K_u2 = 1.0/math.sqrt(2.0)*(b_ee-b_gg)*np.identity(4)-g1/(2.0*math.sqrt(2))*(b_ee*L1@L1d-b_gg*L1d@L1)\
                -g2/(2.0*math.sqrt(2))*(b_ee*L2@L2d-b_gg*L2d@L2)-math.sqrt(0.5*g1*g2)*(b_gg*L1@L2-b_ee*L1d@L2d)
        K_u3 = -1.0j/math.sqrt(2.0)*(math.sqrt(g1)*(b_gg*L1+b_ee*L1d) + math.sqrt(g2)*(b_gg*L2+b_ee*L2d))
        K_u4 = -1.0j/math.sqrt(2.0)*(math.sqrt(g1)*(b_gg*L1-b_ee*L1d) - math.sqrt(g2)*(b_gg*L2-b_ee*L2d))
        
    return (K_u1,K_u2,K_u3,K_u4)
 

# -------------------------------------------------------------------------------------
#                          Completeness relation for Kraus operators
# -------------------------------------------------------------------------------------
def completeness(name):
    result = np.transpose(np.conj(Kraus_opt(name)[0]))@Kraus_opt(name)[0]+\
             np.transpose(np.conj(Kraus_opt(name)[1]))@Kraus_opt(name)[1]+\
             np.transpose(np.conj(Kraus_opt(name)[2]))@Kraus_opt(name)[2]+\
             np.transpose(np.conj(Kraus_opt(name)[3]))@Kraus_opt(name)[3]
                
    return result





