import numpy as np
import math, cmath
from scipy import linalg as LA


def mixed_fidelity(rho1,rho2):
    '''This functions calculates state fidelity between two state matrices.'''
    w1,v1 = LA.eig(rho1) # diagonalising rho1
    w,v = LA.eig(v1@np.diag(np.sqrt(w1))@LA.inv(v1)@rho2@v1@np.diag(np.sqrt(w1))@LA.inv(v1)) #diagonalising the term inside the square root
    argtr = v@np.diag(np.sqrt(w))@LA.inv(v) # Trace argument
    result = np.real((np.trace(argtr))**2)
    return result

def log_negativity(rho):
    '''This function calculates logarithmic negativity of the state rho for two qubits. '''
    n1 = rho.shape[0]//2
    n2 = rho.shape[0]//2
    rho_pt  = np.zeros((n1*n2,n1*n2),dtype=np.complex128)
    for k in range(n1):
        for i in range(n1):
            rho_pt[k*n2:(k+1)*n2,i*n2:(i+1)*n2] = rho[k*n2:(k+1)*n2,i*n2:(i+1)*n2].T
                
    w,v = LA.eigh(rho_pt.conj().T@rho_pt)
    eig_sum = 0.0
    for m in range(np.size(w)):
        if w[m]<0:
            w[m]=0.0
        eig_sum += math.sqrt(w[m])
                
    if eig_sum<1:
        eig_sum = 1.0
            
    result = np.real(math.log2(eig_sum))
    return result
    


def two_qb_cor(rho):
    '''This function calculates 2-qubit corrleation of the state. Ref: PRL 115,220501 (2015). '''
    # Pauli matrices
    s = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]])
    t_sqrd = 0.0
    for i in range(s.ndim):
        for k in range(s.ndim):
            t_sqrd += (np.trace(rho@np.kron(s[i],s[k])))**2
                
    result = 0.25*(1.0+t_sqrd)
    return result
    
def two_qb_bloch_vec(rho):
    '''This function calculates the Bloch vectors of a 2-qubit system. Ref: PRA 93,062320 (2016). '''
    x1 =  2*np.real(rho[0,2]+rho[1,3])
    y1 = -2*np.imag(rho[0,2]+rho[1,3])
    z1 = np.real(rho[0,0]+rho[1,1]-rho[2,2]-rho[3,3])
    rho_a = np.array([x1,y1,z1])
        
    x2 =  2*np.real(rho[0,1]+rho[2,3])
    y2 = -2*np.imag(rho[0,1]+rho[2,3])
    z2 = np.real(rho[0,0]-rho[1,1]+rho[2,2]-rho[3,3])
    rho_b = np.array([x2,y2,z2])
    return (rho_a,rho_b)

def two_qb_bloch_cor(rho):
    '''This function calculates the Bloch vector coordinates of a 2-qubit system. Ref: PRA 93,062320 (2016). '''
    x1 =  2*np.real(rho[0,2]+rho[1,3])
    y1 = -2*np.imag(rho[0,2]+rho[1,3])
    z1 = np.real(rho[0,0]+rho[1,1]-rho[2,2]-rho[3,3])
        
    x2 =  2*np.real(rho[0,1]+rho[2,3])
    y2 = -2*np.imag(rho[0,1]+rho[2,3])
    z2 = np.real(rho[0,0]-rho[1,1]+rho[2,2]-rho[3,3])
    return (x1,y1,z1,x2,y2,z2)
