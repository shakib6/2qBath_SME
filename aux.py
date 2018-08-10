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
#             Joint measurement basis, Eq.(45) August 2018 draft
# -------------------------------------------------------------------------------------
def joint_maps(rho_c):
    rho_c_u1 = (alpha_plus + beta_plus)*rho_c - 0.5*g1*(alpha_plus*rho_c@L1@L1d+beta_plus*rho_c@L1d@L1 + \
                                       np.conj(alpha_plus)*L1@L1d@rho_c+np.conj(beta_plus)*L1d@L1@rho_c)+\
                                              - 0.5*g2*(alpha_plus*rho_c@L2@L2d+beta_plus*rho_c@L2d@L2 + \
                                       np.conj(alpha_plus)*L2@L2d@rho_c+np.conj(beta_plus)*L2d@L2@rho_c)+\
                                    math.sqrt(g1*g2)*(alpha_plus*rho_c@L1@L2+beta_plus*rho_c@L1d@L2d +\
                                        np.conj(alpha_plus)*L2d@L1d@rho_c+np.conj(beta_plus)*L2@L1@rho_c)
    rho_c_u2 = (alpha_minus - beta_minus)*rho_c - 0.5*g1*(alpha_minus*rho_c@L1@L1d-beta_minus*rho_c@L1d@L1 + \
                                       np.conj(alpha_minus)*L1@L1d@rho_c-np.conj(beta_minus)*L1d@L1@rho_c)+\
                                                - 0.5*g2*(alpha_minus*rho_c@L2@L2d-beta_minus*rho_c@L2d@L2 + \
                                       np.conj(alpha_minus)*L2@L2d@rho_c-np.conj(beta_minus)*L2d@L2@rho_c)+\
                                    math.sqrt(g1*g2)*(alpha_minus*rho_c@L1@L2-beta_minus*rho_c@L1d@L2d +\
                                        np.conj(alpha_minus)*L2d@L1d@rho_c-np.conj(beta_minus)*L2@L1@rho_c)
    rho_c_u3 = g1*(b_gg*L1+b_ee*L1d)@rho_c@(np.conj(b_gg)*L1d+np.conj(b_ee)*L1) + \     
               g2*(b_gg*L2+b_ee*L2d)@rho_c@(np.conj(b_gg)*L2d+np.conj(b_ee)*L2) + \
               math.sqrt(g1*g2)*((b_gg*L1+b_ee*L1d)@rho_c@(np.conj(b_gg)*L2d+np.conj(b_ee)*L2) +\
                                 (b_gg*L2+b_ee*L2d)@rho_c@(np.conj(b_gg)*L1d+np.conj(b_ee)*L1))
    rho_c_u4 = g1*(b_gg*L1-b_ee*L1d)@rho_c@(np.conj(b_gg)*L1d-np.conj(b_ee)*L1) + \     
               g2*(b_gg*L2-b_ee*L2d)@rho_c@(np.conj(b_gg)*L2d-np.conj(b_ee)*L2) - \
               math.sqrt(g1*g2)*((b_gg*L1-b_ee*L1d)@rho_c@(np.conj(b_gg)*L2d-np.conj(b_ee)*L2) +\
                                 (b_gg*L2-b_ee*L2d)@rho_c@(np.conj(b_gg)*L1d-np.conj(b_ee)*L1))
    return (rho_c_u1,rho_c_u2,rho_c_u3,rho_c_u4)
            
            
            
            


