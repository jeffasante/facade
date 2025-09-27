'''
Simple test script to verify JAX is using MPS/GPU if available, otherwise CPU.
Also runs a quick benchmark to show JAX performance.
'''

# Device configuration for MPS/GPU/CPU
import jax
import jax.numpy as jnp

# Detect available devices
devices = jax.devices()
print("Available JAX devices:", devices)

# Check for GPU (either 'gpu' or 'cuda' in the name)
if any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices):
    print("GPU backend detected and will be used by JAX.")

# Check for Apple Silicon
elif any("mps" in str(d).lower() for d in devices):
    print("MPS backend (Apple Silicon) detected and will be used by JAX.")

else:
    print("No accelerator found. JAX will use the CPU backend.")

# You can still check the default backend JAX has chosen
print("JAX default backend:", jax.default_backend())


def quick_benchmark():
    """Quick benchmark to show JAX performance"""
    print("Running quick JAX benchmark...")

    # Create some sample data
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (1000, 100))
    w = jax.random.normal(key, (100, 50))

    # Time matrix multiplication
    import time
    start = time.time()
    for _ in range(100):
        result = jnp.dot(x, w)
    elapsed = time.time() - start

    print(f"⚡ 100 matrix multiplications: {elapsed:.3f}s")
    print(f" Operations per second: {100/elapsed:.1f}")

# Run benchmark
quick_benchmark()