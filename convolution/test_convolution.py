import ctypes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from pathlib import Path
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
LIB_PATH = SCRIPT_DIR / "libconvolution.so"

if not LIB_PATH.exists():
    raise FileNotFoundError(
        f"{LIB_PATH} not found. Compile it with: "
        "nvcc -arch=sm_75 -Xcompiler -fPIC -shared "
        "convolution/convolution_lib.cu -o convolution/libconvolution.so"
    )

lib = ctypes.cdll.LoadLibrary(str(LIB_PATH))

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
    image_path = SCRIPT_DIR / "test_image.jpg"

    if not image_path.exists():
        request = Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=30) as response:
                image_path.write_bytes(response.read())
        except Exception as exc:
            print(f"Image download failed ({exc}). Using random noise.")
            return np.random.rand(512, 512).astype(np.float32)
    
    try:
        img_raw = mpimg.imread(str(image_path))
    except FileNotFoundError:
        print("Image download failed. Using random noise.")
        return np.random.rand(512, 512).astype(np.float32)

    # Convert RGB to Grayscale
    if len(img_raw.shape) == 3:
        # Standard luminance formula
        img = np.dot(img_raw[...,:3], [0.299, 0.587, 0.114])
    else:
        img = img_raw
    
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img

def make_box_filter(filter_size):
    filter_val = 1.0 / (filter_size * filter_size)
    return np.full((filter_size, filter_size), filter_val, dtype=np.float32)

def make_sobel_x_filter():
    return np.array(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        dtype=np.float32
    )

def run_experiment(image_base, tile_scale, conv_filter, filter_label):
    """
    Runs a single convolution experiment with specified image scaling and filter size.
    Returns: Width, Height, GPU Time, CPU Time, Input Image, GPU Output, CPU Output
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
    
    conv_filter = np.asarray(conv_filter, dtype=np.float32)
    if conv_filter.ndim != 2 or conv_filter.shape[0] != conv_filter.shape[1]:
        raise ValueError("The convolution filter must be a square 2D array.")
    conv_filter = np.ascontiguousarray(conv_filter, dtype=np.float32)
    filter_size = conv_filter.shape[0]

    # Prepare Output Buffers
    out_gpu = np.zeros_like(image, dtype=np.float32)
    out_cpu = np.zeros_like(image, dtype=np.float32)

    print(f"Testing Image: {W}x{H} | Filter: {filter_label}...")

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

    return W, H, gpu_time, cpu_time, image, out_gpu, out_cpu

def main():
    img_base = get_test_image()
    results = []

    print("=== STARTING MULTI-SIZE PERFORMANCE TEST ===")
    
    # Experiment A: Varying Image Size (N) with fixed Sobel edge filter (3x3)
    # Scales: 1x1 (Small), 3x3 (Medium), 5x5 (Large)
    scales = [1, 3, 5]
    sobel_filter = make_sobel_x_filter()
    demo_input = None
    sobel_gpu_demo = None
    sobel_cpu_demo = None

    for s in scales:
        w, h, gt, ct, image, out_gpu, out_cpu = run_experiment(
            img_base,
            tile_scale=s,
            conv_filter=sobel_filter,
            filter_label="Sobel X 3x3"
        )
        if s == 1:
            demo_input = image
            sobel_gpu_demo = out_gpu
            sobel_cpu_demo = out_cpu

        results.append({
            "Type": "Varying Image Size",
            "Image": f"{w}x{h}",
            "Filter": "Sobel 3x3",
            "GPU Time": gt,
            "CPU Time": ct,
            "Speedup": ct / gt if gt > 0 else 0
        })

    # Experiment B: Varying Filter Size (M) with fixed image size.
    # Box filters make it easy to compare 5x5 and 7x7 stencil cost.
    filter_sizes = [5, 7]
    for f in filter_sizes:
        box_filter = make_box_filter(f)
        w, h, gt, ct, image, out_gpu, out_cpu = run_experiment(
            img_base,
            tile_scale=3,
            conv_filter=box_filter,
            filter_label=f"Box Blur {f}x{f}"
        )
        results.append({
            "Type": "Varying Filter Size",
            "Image": f"{w}x{h}",
            "Filter": f"Box {f}x{f}",
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

    # --- Save Visual Results ---
    RESULTS_DIR.mkdir(exist_ok=True)
    cpu_edge_output = np.abs(sobel_cpu_demo)
    gpu_edge_output = np.abs(sobel_gpu_demo)

    edge_path = RESULTS_DIR / "edge_detection_result.png"
    demo_path = RESULTS_DIR / "convolution_output_demo.png"
    comparison_path = RESULTS_DIR / "comparison.png"

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(demo_input, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(cpu_edge_output, cmap="gray")
    plt.title("CPU Sobel")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(gpu_edge_output, cmap="gray")
    plt.title("CUDA Sobel")
    plt.axis("off")
    plt.savefig(edge_path, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.imshow(gpu_edge_output, cmap="gray")
    plt.title("CUDA Sobel Edge Detection Output")
    plt.axis("off")
    plt.savefig(demo_path, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(demo_input, cmap="gray")
    plt.title("Input Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(gpu_edge_output, cmap="gray")
    plt.title("CUDA Sobel Output")
    plt.axis("off")
    plt.savefig(comparison_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\nVisual results saved under '{RESULTS_DIR}'")

if __name__ == "__main__":
    main()
