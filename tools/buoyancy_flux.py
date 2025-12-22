from scipy import interpolate

Cp = 4000 # J kg-1 K-1

# Values from Table A3.1 on page 603 of the book "Atmosphere-Ocean Dynamics" by A.E. Gill
T = [-2, 0, 2, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31] # deg C
alpha = [254, 526, 781, 1021, 1357, 1668, 1958, 2230, 2489, 2734, 2970, 3196, 3413] # 10**-7 K-1
drho_dS = [0.814, 0.808, 0.801, 0.796, 0.788, 0.781, 0.775, 0.769, 0.764, 0.760, 0.756, 0.752, 0.749]
f_interp_alpha = interpolate.interp1d(T, alpha, fill_value='extrapolate')
f_interp_beta = interpolate.interp1d(T, drho_dS, fill_value='extrapolate')

def get_alpha_at_surface(temp:float) -> float: # thermal expansion coefficient under p = 0 bar, and S = 35
    return f_interp_alpha(temp)*10**(-7)

def get_beta_at_surface(temp:float) -> float: # haline contraction coefficient under p = 0 bar, and S = 35
    # beta = 1/rho * drho/dS but this is multiplied by rho to calculate buoyancy salt flux, so ignoring 1/rho
    return f_interp_beta(temp)

def calculate_buoyancy_heat_flux(qnet:float, temp:float) -> float:
    alpha = get_alpha_at_surface(temp)
    Bh = alpha * qnet/Cp
    return Bh

def calculate_buoyancy_salt_flux(sflux:float, temp:float) -> float:
    beta_rho = get_beta_at_surface(temp) # beta/rho
    Bw = -beta_rho*sflux
    return Bw

def calculate_buoyancy_flux(qnet:float, sflux:float, temp:float) -> float:
    Bh = calculate_buoyancy_heat_flux(qnet, temp)
    Bw = calculate_buoyancy_salt_flux(sflux, temp)
    B = Bh + Bw
    return B # kg s-1 m-2
