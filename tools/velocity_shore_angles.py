import numpy as np

def get_cross_and_along_shelf_velocities(h:np.ndarray[float],
                                         u:np.ndarray[float],
                                         v:np.ndarray[float]) -> tuple[np.ndarray, np.ndarray]:
    dh_dy, dh_dx = np.gradient(h)
    # normal vector (across bathymetry, positive towards larger h)
    grad_mag = np.sqrt(dh_dx**2 + dh_dy**2)
    nx = dh_dx / grad_mag
    ny = dh_dy / grad_mag
    # tangent vector
    tx = ny
    ty = -nx

    u_across = u * nx + v * ny
    u_along = u * tx + v * ty
    return u_across, u_along
