import ctypes
import numpy as np
import time

# 1. Load the shared library
lib = ctypes.cdll.LoadLibrary("./libmatrix.so")

# 2. Define the argument types for the C function
# void gpu_matrix_multiply(float *A, float *B, float *C, int N)
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

def run_test(N):
    print(f"Testing N={N} with Python + CUDA...")
    
    # Generate random matrices (Float32)
    # Note: We flatten them to 1D arrays because C expects pointers
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    # Measure time
    start = time.time()
    
    # Call the C function from the library
    lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
    
    end = time.time()
    print(f"-> Done in {end - start:.4f} seconds\n")
    return end - start

# Run tests
if __name__ == "__main__":
    run_test(1024)
    run_test(2048)
    run_test(4096)