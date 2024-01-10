import numpy as np

# The functions in this script are based on the "seawater toolbox"
# created for Matlab by Phil Morgan and Lindsay Pender at CSIRO.
#
# The equations used are those as defined in the UNESCO 1983
# "Algorithms for computation of fundamental properties of seawater",
# published in the Unesco Technical Papers in Marine Science, No. 44.
#
# The relevant equation numbers from this document are referenced
# in the functions when applicable.
#
# Symbols/abbreviations used:
# T: temperature
# S: salinity
# P: pressure
# rho: density
# K: secant bulk modulus

def calculate_density_of_standard_mean_ocean_water(T:np.ndarray) -> np.ndarray:
    '''Calculates the density of reference pure water.
    Equation 14, page 17 of Unesco 1983.'''

    a0 = 999.842594
    a1 = 6.793952e-2
    a2 = -9.095290e-3
    a3 = 1.001685e-4
    a4 = -1.120083e-6
    a5 = 6.536332e-9

    T68 = T*1.00024
    rho_smow = a0+a1*T+a2*T**2+a3*T**3+a4*T**4+a5*T**5
    
    return rho_smow

def calculate_density_at_atmospheric_pressure(S:np.ndarray, T:np.ndarray) -> np.ndarray:
    '''Calculates the density of sea water at atmospheric pressure (P=0).
    Equation 13, page 17 of Unesco 1983.'''

    b0 =  8.24493e-1
    b1 = -4.0899e-3
    b2 =  7.6438e-5
    b3 = -8.2467e-7
    b4 =  5.3875e-9

    c0 = -5.72466e-3
    c1 = +1.0227e-4
    c2 = -1.6546e-6

    d0 = 4.8314e-4

    T68 = T*1.00024
    rho_smow = calculate_density_of_standard_mean_ocean_water(T)
    rho_P0 = rho_smow+(b0+b1*T+b2*T**2+b3*T**3+b4*T**4)*S+(c0+c1*T+c2*T**2)*S**(3/2)+d0*S**2

    return rho_P0

def calculate_secant_bulk_modulus(S:np.ndarray, T:np.ndarray, P:np.ndarray):
    '''Calculate secant bulk modulus of sea water.
    Equations 15 to 19, pages 18-19 of Unesco 1983.'''

    e4 = -5.155288E-5
    e3 = +1.360477E-2
    e2 = -2.327105
    e1 = +148.4206
    e0 = 19652.21

    f3 =  -6.1670E-5
    f2 =  +1.09987E-2
    f1 =  -0.603459
    f0 = +54.6746

    g2 = -5.3009E-4
    g1 = +1.6483E-2
    g0 = +7.944E-2

    h3 = -5.77905E-7
    h2 = +1.16092E-4
    h1 = +1.43713E-3
    h0 = +3.239908

    i2 = -1.6078E-6
    i1 = -1.0981E-5
    i0 =  2.2838E-3

    j0 = 1.91075E-4

    k2 =  5.2787E-8
    k1 = -6.12293E-6
    k0 =  +8.50935E-5

    m2 =  9.1697E-10
    m1 = +2.0816E-8
    m0 = -9.9348E-7

    K_w = e0+e1*T+e2*T**2+e3*T**3+e4*T**4 # pure water term: equation 19
    K_P0 = K_w+(f0+f1*T+f2*T**2+f3*T**3)*S+(g0+g1*T+g2*T**2)*S**(3/2) # at atmospheric pressure: equation 16
    
    A_w = h0+h1*T+h2*T**2+h3*T**3
    A = A_w+(i0+i1*T+i2*T**2)*S+j0*S**(3/2) # equation 17

    B_w = k0+k1*T+k2*T**2
    B = B_w+(m0+m1*T+m2*T**2)*S # equation 18
    
    K = K_P0+A*P+B*P**2 # secant bulk modulus: equation 15

    return K

def calculate_density(S:np.ndarray, T:np.ndarray, P:np.ndarray) -> np.ndarray:
    '''Calculates the density of sea water.
    Equation 7, page 15 of Unesco 1983.'''

    rho_P0 = calculate_density_at_atmospheric_pressure(S, T)
    K = calculate_secant_bulk_modulus(S, T, P)
    P = P/10. # convert from db to atm pressure units
    
    rho = rho_P0/(1-P/K)

    return rho
