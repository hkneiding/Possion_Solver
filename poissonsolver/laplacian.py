from numpy import zeros, float64
from scipy.linalg import block_diag


def create_laplacian_1d(n_grid_x, h):
    fx = 1/h[0] ** 2
    laplacian = zeros(shape = (n_grid_x, n_grid_x), dtype = float64)
    
    for i in range(n_grid_x):
        laplacian[i, i] = -2 * fx
        laplacian[i, (i + 1) % n_grid_x] += 1.0 * fx
        laplacian[i, (i - 1) % n_grid_x] += 1.0 * fx
    
    return laplacian

def create_laplacian_2d(n_grid_x, n_grid_y, h):
    fx = 1/h[0] ** 2
    fy = 1/h[1] ** 2
    fxy = fx + fy
    n_grid = n_grid_x * n_grid_y
    
    B = np.array(create_laplacian_1d(n_grid_x, h))
    A = sp.linalg.block_diag(B)
    for i in range(n_grid_y - 1):        
        A = sp.linalg.block_diag(A,B)
    laplacian = A
        
    for i in range(n_grid):
        laplacian[i ,i] = -2 * fxy
        laplacian[i, (i + n_grid_x) % n_grid] += 1.0 * fy
        laplacian[i, (i - n_grid_x) % n_grid] += 1.0 * fy

    return laplacian

def create_laplacian_3d(n_grid_x, n_grid_y, n_grid_z, h):
    fx = 1/h[0] ** 2
    fy = 1/h[1] ** 2
    fz = 1/h[2] ** 2
    fxyz = fx + fy + fz
    n_grid = n_grid_x  * n_grid_y * n_grid_z 

    B = np.array(create_laplacian_2d(n_grid_x, n_grid_y, h))
    A = sp.linalg.block_diag(B)
    for i in range(n_grid_z - 1):        
        A = sp.linalg.block_diag(A,B)
    laplacian = A
    
    for i in range(n_grid):
        laplacian[i ,i] = -2 * fxyz
        laplacian[i, (i + (n_grid_x * n_grid_y)) % n_grid] += 1.0 * fz
        laplacian[i, (i - (n_grid_x * n_grid_y)) % n_grid] += 1.0 * fz

    return laplacian
    
    
def create_laplacian(*n_grid, h):
    if len(n_grid) == 1:
        return create_laplacian_1d(*n_grid, h)
    elif len(n_grid) == 2:
        return create_laplacian_2d(*n_grid, h)
    elif len(n_grid) == 3:
        return create_laplacian_3d(*n_grid, h)
    else:
        raise ValueError('create_laplacian expects 1, 2 or 3')
       
