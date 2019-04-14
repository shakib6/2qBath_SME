# ------------------------------------------------------------------------
#                                Input data 
# ------------------------------------------------------------------------
import numpy as np
import math, cmath
from scipy import linalg as LA



# --- Initial system state ---
a = 0.0
b = math.sqrt(1.0-abs(a)**2)

# --- Environment's state ---
epsilon = 0.5
b_ee = math.sqrt(1.0/(2.0+epsilon)) + 0j
b_gg = math.sqrt((1.0+epsilon)/(2.0+epsilon)) + 0j
b_eg = math.sqrt(0.0/4.0) + 0j
b_ge = math.sqrt(0.0/4.0) + 0j

if abs(b_ee)**2 + abs(b_eg)**2 + abs(b_gg)**2 + abs(b_ge)**2 < 0.99999:
    print('>>>>> The state of the bath is not normalised.')
    
# --- Coupling strengths ---
lambda1, lambda2 = 1.0e+1, 1.0e+1

# --- time step ---
dt = 1.0e-3
counter = round(1000.0/dt)
ntraj = 1

# --- Dimensionless gammas ---
g1 = (lambda1*dt)**2
g2 = (lambda2*dt)**2

# --- which maps ---
# 1 = local_maps, 2 = joint_maps, 3 = energy-space_map
ch = 3
name = 'energy'

# --- joint measurement parameters ---
alpha_plus  = np.conj(b_ee)*(b_ee + b_gg)
alpha_minus = np.conj(b_ee)*(b_ee - b_gg)
beta_plus   = np.conj(b_gg)*(b_ee + b_gg)
beta_minus  = np.conj(b_gg)*(b_ee - b_gg)


# --- defining lowering (sigma) and rasing operators (sigma^\dagger) ---
s = np.zeros((2,2),dtype=np.float64)
s[1,0]=1

sd = s.conj().T

iden = np.identity(2)

# --- Lindblad operators ---
L1  = np.kron(s,iden)
L2  = np.kron(iden,s)
L1d = np.kron(sd,iden)
L2d = np.kron(iden,sd)


# --- Conditional density matrices ---
rho_c1 = np.zeros((2,2),dtype=np.complex128)
rho_c2 = np.zeros((2,2),dtype=np.complex128)
rho_c = np.kron(rho_c1,rho_c2)
rho_c_u1 = np.zeros((4,4),dtype=np.complex128)
rho_c_u2 = np.zeros((4,4),dtype=np.complex128)
rho_c_u3 = np.zeros((4,4),dtype=np.complex128)
rho_c_u4 = np.zeros((4,4),dtype=np.complex128)

# --- Joint system S.S. ---
rho_ss = np.zeros((4,4),dtype=np.complex128)
rho_eq = np.zeros((4,4,ntraj),dtype=np.complex128)

log_neg_c = np.zeros((ntraj,counter+1),dtype=np.complex128)
# tq_cor  = np.zeros(counter+1,dtype=np.complex128)
purity_c  = np.zeros((ntraj,counter+1),dtype=np.complex128)
fid_c  = np.zeros((ntraj,counter+1),dtype=np.complex128)


# -----------------------------
#         Bloch vectors  
# -----------------------------
x1_c = np.zeros((ntraj,counter+1),dtype=np.float64)
y1_c = np.zeros((ntraj,counter+1),dtype=np.float64)
z1_c = np.zeros((ntraj,counter+1),dtype=np.float64)
x2_c = np.zeros((ntraj,counter+1),dtype=np.float64)
y2_c = np.zeros((ntraj,counter+1),dtype=np.float64)
z2_c = np.zeros((ntraj,counter+1),dtype=np.float64)


