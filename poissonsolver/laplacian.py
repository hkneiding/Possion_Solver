from numpy import zeros, float64

def create_laplacian_1d(n_grid_x):
  laplacian = zeros(shape = (n_grid_x, n_grid_x), dtype = float64)
  for i in range(n_grid_x):
    laplacian[i, i] = -2.0
    laplacian[i, (i + 1) % n_grid_x] += 1.0
    laplacian[i, (i - 1) % n_grid_x] += 1.0
    

def create_laplacian(*n_grid, h):
  if len(n_grid) == 1:
    return create_laplacian_1d(*n_grid)
  elif len(n_grid) == 2:
    return create_laplacian_2d(*n_grid, h)
  elif len(n_grid) == 3:
    return create_laplacian_3d(*n_grid, h)
  else:
    raise ValueError('create_laplacian expects 1, 2 or 3')
