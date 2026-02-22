"""
Quick script to check TensorFlow GPU availability
"""

import sys

try:
    import tensorflow as tf

    print("=" * 60)
    print("TensorFlow GPU Check")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

    # List all devices
    print("\nAll available devices:")
    for device in tf.config.list_physical_devices():
        print(f"  - {device}")

    # Check GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✅ GPU ENABLED - Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        print("\n⚡ PINN training will be accelerated!")
    else:
        print("\n⚠️  No GPU detected - using CPU")
        print("   PINN training will be slower (~10-20x)")

    # Quick benchmark
    print("\nRunning quick benchmark...")
    import time

    # Small matrix multiplication test
    with tf.device('/CPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        start = time.time()
        c = tf.matmul(a, b)
        cpu_time = time.time() - start
        print(f"  CPU time: {cpu_time*1000:.2f} ms")

    if gpus:
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            start = time.time()
            c = tf.matmul(a, b)
            gpu_time = time.time() - start
            print(f"  GPU time: {gpu_time*1000:.2f} ms")
            print(f"  Speedup: {cpu_time/gpu_time:.1f}x faster")

    print("=" * 60)

except ImportError:
    print("❌ TensorFlow not installed")
    sys.exit(1)
