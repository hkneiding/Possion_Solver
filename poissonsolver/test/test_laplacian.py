import numpy as np
import numpy.testing as npt

from ..laplacian import create_laplacian

def test_create_laplacian_1d_2():
    npt.assert_equal(
        create_laplacian(2,h = [1]),
        np.asarray([[-2.0, 2.0], 
                    [2.0, -2.0]], dtype=np.float64))

def test_create_laplacian_1d_3():
    npt.assert_equal(
        create_laplacian(3, h = [1]),
        np.asarray([
            [-2.0, 1.0, 1.0],
            [1.0, -2.0, 1.0],
            [1.0, 1.0, -2.0]], dtype=np.float64))

def test_create_laplacian_1d_4():
    npt.assert_equal(
        create_laplacian(4, h = [1]),
        np.asarray([
            [-2.0, 1.0, 0.0, 1.0],
            [1.0, -2.0, 1.0, 0.0],
            [0.0, 1.0, -2.0, 1.0],
            [1.0, 0.0, 1.0, -2.0]], dtype=np.float64))

def test_create_laplacian_1d_2_1():
    npt.assert_array_almost_equal(
        create_laplacian(2, h = [0.1]),
        np.asarray([[-200.0, 200.0], 
                    [200.0, -200.0]], dtype=np.float64))
    

def test_create_laplacian_2d_2x2():
    npt.assert_array_almost_equal(
        create_laplacian(2,2, h = [1,1]),
        np.asarray([
            [-4.0, 2.0, 2.0, 0.0],
            [2.0, -4.0, 0.0, 2.0],
            [2.0, 0.0, -4.0, 2.0],
            [0.0, 2.0, 2.0, -4.0]], dtype=np.float64))

def test_create_laplacian_2d_3x2():
    npt.assert_array_almost_equal(
        create_laplacian(3,2, h = [1,1]),
        np.asarray([
            [-4.0, 1.0, 1.0, 2.0, 0.0, 0.0],
            [1.0, -4.0, 1.0, 0.0, 2.0, 0.0],
            [1.0, 1.0, -4.0, 0.0, 0.0, 2.0],
            [2.0, 0.0, 0.0, -4.0, 1.0, 1.0],
            [0.0, 2.0, 0.0, 1.0, -4.0, 1.0],
            [0.0, 0.0, 2.0, 1.0, 1.0, -4.0]], dtype=np.float64))

def test_create_laplacian_2d_2x3():
    npt.assert_array_almost_equal(
        create_laplacian(2,3, h = [1,1]),
        np.asarray([
            [-4.0, 2.0, 1.0, 0.0, 1.0, 0.0],
            [2.0, -4.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, -4.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, -4.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, -4.0, 2.0],
            [0.0, 1.0, 0.0, 1.0, 2.0, -4.0]], dtype=np.float64))
    
def test_create_laplacian_2d_3x3():
    npt.assert_array_almost_equal(
        create_laplacian(3,3, h = [1,1]),
        np.asarray([
            [-4.0,  1.0,  1.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0],
            [ 1.0, -4.0,  1.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0],
            [ 1.0,  1.0, -4.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0],
            [ 1.0,  0.0,  0.0, -4.0,  1.0,  1.0,  1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0,  1.0, -4.0,  1.0,  0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0,  1.0,  1.0, -4.0,  0.0,  0.0,  1.0],
            [ 1.0,  0.0,  0.0,  1.0,  0.0,  0.0, -4.0,  1.0,  1.0],
            [ 0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  1.0, -4.0,  1.0],
            [ 0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  1.0,  1.0, -4.0]], dtype=np.float64))
    

def test_create_laplacian_2d_2x2_1():
    npt.assert_array_almost_equal(
        create_laplacian(2,2, h = [0.1,0.1]),
        np.asarray([
            [-400.0, 200.0, 200.0, 0.0],
            [200.0, -400.0, 0.0, 200.0],
            [200.0, 0.0, -400.0, 200.0],
            [0.0, 200.0, 200.0, -400.0]], dtype=np.float64))
