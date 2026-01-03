import ctypes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os

# Load the Shared Library
lib = ctypes.cdll.LoadLibrary("./libconvolution.so")

# Define argument types for the C functions (GPU and CPU)
common_argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), # Input Image
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), # Filter
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"), # Output
    ctypes.c_int, # Width
    ctypes.c_int, # Height
    ctypes.c_int  # Filter Size
]
lib.gpu_convolution.argtypes = common_argtypes
lib.cpu_convolution.argtypes = common_argtypes

def get_test_image():
    """
    Downloads and prepares a test image (Taj Mahal).
    Returns a normalized grayscale NumPy array (float32).
    """
    img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Taj-Mahal.jpg/800px-Taj-Mahal.jpg"
    if not os.path.exists("test_image.jpg"):
        # Use a User-Agent to avoid 403 Forbidden errors
        os.system(f"wget -U 'Mozilla/5.0' -O test_image.jpg {img_url}")
    
    try:
        img_raw = mpimg.imread("test_image.jpg")
    except FileNotFoundError:
        print("Image download failed. Using random noise.")
        return np.random.rand(512, 512).astype(np.float32)

    # Convert RGB to Grayscale
    if len(img_raw.shape) == 3:
        # Standard luminance formula
        img = np.dot(img_raw[...,:3], [0.299, 0.587, 0.114])
    else:
        img = img_raw
    
    # Normalize to [0, 1] range
    return (img.astype(np.float32) / 255.0)

def run_experiment(image_base, tile_scale, filter_size):
    """
    Runs a single convolution experiment with specified image scaling and filter size.
    Returns: Width, Height, GPU Time, CPU Time, Output Image
    """
    # Prepare Image (Varying N)
    # Scale up the image by tiling it (Mosaic)
    if tile_scale > 1:
        image = np.tile(image_base, (tile_scale, tile_scale))
    else:
        image = image_base.copy()
    
    # Force contiguous memory layout for C-compatibility
    image = np.ascontiguousarray(image, dtype=np.float32)
    H, W = image.shape
    
    # Prepare Filter (Varying M)
    # Using Box Blur filter: all elements are 1/(size^2)
    # This allows us to easily change filter size (3x3, 5x5, 7x7)
    filter_val = 1.0 / (filter_size * filter_size)
    conv_filter = np.full((filter_size, filter_size), filter_val, dtype=np.float32)
    conv_filter = np.ascontiguousarray(conv_filter)

    # Prepare Output Buffers
    out_gpu = np.zeros_like(image, dtype=np.float32)
    out_cpu = np.zeros_like(image, dtype=np.float32)

    print(f"Testing Image: {W}x{H} | Filter: {filter_size}x{filter_size}...")

    # Run GPU Test
    # Warmup run (initialize context)
    lib.gpu_convolution(image, conv_filter, out_gpu, W, H, filter_size)
    
    start = time.time()
    lib.gpu_convolution(image, conv_filter, out_gpu, W, H, filter_size)
    gpu_time = time.time() - start

    # Run CPU Test
    # For very large images (scale > 3), CPU might take too long.
    # We run it here for completeness as per lab requirements.
    start = time.time()
    lib.cpu_convolution(image, conv_filter, out_cpu, W, H, filter_size)
    cpu_time = time.time() - start

    return W, H, gpu_time, cpu_time, out_gpu

def main():
    img_base = get_test_image()
    results = []

    print("=== STARTING MULTI-SIZE PERFORMANCE TEST ===")
    
    # Experiment A: Varying Image Size (N) with fixed Filter Size (3x3)
    # Scales: 1x1 (Small), 3x3 (Medium), 5x5 (Large)
    scales = [1, 3, 5]
    for s in scales:
        w, h, gt, ct, out = run_experiment(img_base, tile_scale=s, filter_size=3)
        results.append({
            "Type": "Varying Image Size",
            "Image": f"{w}x{h}",
            "Filter": "3x3",
            "GPU Time": gt,
            "CPU Time": ct,
            "Speedup": ct / gt if gt > 0 else 0
        })

    # Experiment B: Varying Filter Size (M) with fixed Image Size (Medium 3x3)
    # Filter Sizes: 5x5, 7x7 (we already did 3x3 in Exp A)
    filter_sizes = [5, 7]
    for f in filter_sizes:
        w, h, gt, ct, out = run_experiment(img_base, tile_scale=3, filter_size=f)
        results.append({
            "Type": "Varying Filter Size",
            "Image": f"{w}x{h}",
            "Filter": f"{f}x{f}",
            "GPU Time": gt,
            "CPU Time": ct,
            "Speedup": ct / gt if gt > 0 else 0
        })

    # --- Print Summary Table ---
    print("\n" + "="*80)
    print(f"{'Experiment Type':<20} | {'Image Size':<15} | {'Filter':<8} | {'GPU (s)':<10} | {'CPU (s)':<10} | {'Speedup':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['Type']:<20} | {r['Image']:<15} | {r['Filter']:<8} | {r['GPU Time']:.4f}     | {r['CPU Time']:.4f}     | {r['Speedup']:.1f}x")
    print("="*80)

    # --- Save Visual Result ---
    # Save the output image from the last run to demonstrate functionality
    plt.figure(figsize=(10, 5))
    plt.imshow(out, cmap='gray') 
    plt.title("Convolution Output (Box Blur Result)")
    plt.axis('off')
    plt.savefig("convolution_output_demo.png")
    print("\nVisual result saved as 'convolution_output_demo.png'")

if __name__ == "__main__":
    main()