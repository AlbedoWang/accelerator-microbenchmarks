import jax
import jax.numpy as jnp
from jax import lax
import time
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
import gc

MATRIX_SIZE = (15000, 15000) 
DTYPE = jnp.bfloat16

SRC_DEVICE = 0
DST_DEVICE = 1

devices = jax.devices()
mesh = Mesh(devices, ("ici",))
jax.set_mesh(mesh)

def run_benchmark(src_device, dst_device, use_salt=True):
    devices = jax.local_devices()
    n_devices = len(devices)
    if n_devices < 2:
        print("Error: Need at least 2 devices to benchmark P2P.")
        return

    print(f"Running on {n_devices} devices.")
    print(f"Transfer: Device {src_device} -> Device {dst_device}")
    print(f"Allocating data ({MATRIX_SIZE})...")
    host_data = np.random.rand(n_devices, *MATRIX_SIZE).astype(np.float32)
    # sharded_data = jax.device_put_sharded(list(host_data), devices)
    sharded_data = jax.device_put(
        jnp.array(host_data), 
        jax.sharding.NamedSharding(mesh, P('ici',))
    )
    
    payload_bytes = np.prod(MATRIX_SIZE) * 2 
    payload_gb = payload_bytes / 1e9
    print(f"Payload Size: {payload_gb:.4f} GB")

    salt_perm = [(2, 4), (4, 2), (3, 5), (5, 3)] if use_salt else []

    ps = P('ici',)
    @jax.shard_map(
        mesh=mesh,
        in_specs=ps,
        out_specs=ps
    )
    def p2p_kernel(x):
        out = lax.ppermute(
            x,
            axis_name='ici',
            perm=[(src_device, dst_device), (dst_device, src_device)] + salt_perm
        )
        return out
    
    p2p_kernel_jit = jax.jit(
        p2p_kernel,
        in_shardings=ps,
        out_shardings=ps
    )

    print("Warming up (compiling)...")
    _ = p2p_kernel_jit(sharded_data).block_until_ready()
    print("Warmup complete.")

    num_iterations = 5
    timings = []

    print(f"Benchmarking over {num_iterations} iterations...")
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        
        result = p2p_kernel_jit(sharded_data)
        result.block_until_ready()
        
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    avg_time = np.mean(timings)
    min_time = np.min(timings)
    
    bandwidth_avg = payload_gb / avg_time
    bandwidth_peak = payload_gb / min_time

    print("-" * 30)
    print(f"Avg Time:      {avg_time*1000:.2f} ms")
    print(f"Avg Bandwidth: {bandwidth_avg:.2f} GB/s")
    print(f"Peak Bandwidth: {bandwidth_peak:.2f} GB/s")
    print("-" * 30)

    hlo = p2p_kernel_jit.lower(sharded_data).compiler_ir(dialect="hlo")
    with open(f"p2p_hlo_src{src_device}_dst{dst_device}.txt", "w") as f:
        f.write(hlo.as_hlo_text())

if __name__ == "__main__":
    # for dst in range(len(devices)):
    for dst in [1, 6, 7]:
        run_benchmark(SRC_DEVICE, dst, use_salt=True)
    
    gc.collect()
    print("" + "="*50)
    print("Running benchmarks without salt permutations")
    print("="*50 + "\n")

    for dst in [1, 6, 7]:
        run_benchmark(SRC_DEVICE, dst, use_salt=False)