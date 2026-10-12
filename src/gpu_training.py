"""Small dense network trained with OpenCL forward/backprop and Adam on Intel GPU."""
import os
from pathlib import Path
import numpy as np

KERNEL = r"""
__kernel void forward(__global const float *x, __global const float *w,
 __global float *out, int ni, int no, int relu) {
 int row=get_global_id(0), j=get_global_id(1);
 float v=w[j*(ni+1)+ni];
 for(int k=0;k<ni;k++) v+=x[row*ni+k]*w[j*(ni+1)+k];
 out[row*no+j]=relu ? fmax(v,0.0f) : v;
}
__kernel void output_delta(__global const float *pred, __global const float *y,
 __global float *delta) {
 int i=get_global_id(0); delta[i]=pred[i]-y[i];
}
__kernel void backward(__global const float *next_delta, __global const float *next_w,
 __global const float *activation, __global float *delta, int ni, int no) {
 int row=get_global_id(0), j=get_global_id(1); float v=0;
 for(int k=0;k<no;k++) v+=next_delta[row*no+k]*next_w[k*(ni+1)+j];
 delta[row*ni+j]=activation[row*ni+j]>0 ? v : 0;
}
__kernel void update(__global const float *x, __global const float *delta,
 __global float *w, __global float *m, __global float *v, int ni, int n,
 float b1corr, float b2corr) {
 int i=get_global_id(0), j=i/(ni+1), k=i%(ni+1); float g=0;
 for(int row=0;row<n;row++) g+=delta[row*(get_global_size(0)/(ni+1))+j]*(k==ni ? 1.0f : x[row*ni+k]);
 g=g/n + (k==ni ? 0.0f : 0.005f*w[i]);
 m[i]=0.9f*m[i]+0.1f*g; v[i]=0.999f*v[i]+0.001f*g*g;
 w[i]-=0.003f*(m[i]/b1corr)/(sqrt(v[i]/b2corr)+1e-8f);
}
"""


def train_gpu(X, y, epochs=350, seed=42):
    import pyopencl as cl
    cache = Path(__file__).resolve().parents[1] / "outputs/opencl_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYOPENCL_CACHE_DIR", str(cache))
    devices = [d for p in cl.get_platforms() for d in p.get_devices(device_type=cl.device_type.GPU)]
    if not devices:
        raise RuntimeError("No OpenCL GPU exposed by the installed driver.")
    device = next((d for d in devices if "Intel" in d.vendor), devices[0])
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    program = cl.Program(context, KERNEL).build(cache_dir=str(cache))
    forward, output_delta, backward, update = [cl.Kernel(program, name) for name in ["forward", "output_delta", "backward", "update"]]
    flags = cl.mem_flags
    def buffer(array):
        return cl.Buffer(context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(array, dtype=np.float32))
    rng = np.random.default_rng(seed)
    sizes = [4, 32, 16, 2]
    weights = [np.column_stack([rng.normal(0, np.sqrt(2/ni), (no, ni)), np.zeros(no)]).astype(np.float32)
               for ni, no in zip(sizes[:-1], sizes[1:])]
    wb = [buffer(w) for w in weights]
    mb = [buffer(np.zeros_like(w)) for w in weights]
    vb = [buffer(np.zeros_like(w)) for w in weights]
    n = len(X)
    activations = [buffer(X)] + [buffer(np.zeros((n, no))) for no in sizes[1:]]
    deltas = [buffer(np.zeros((n, no))) for no in sizes[1:]]
    target = buffer(y)
    for step in range(1, epochs+1):
        for layer in range(3):
            forward(queue, (n, sizes[layer+1]), None, activations[layer], wb[layer], activations[layer+1],
                    np.int32(sizes[layer]), np.int32(sizes[layer+1]), np.int32(layer < 2))
        output_delta(queue, (n*2,), None, activations[-1], target, deltas[-1])
        for layer in [1, 0]:
            backward(queue, (n, sizes[layer+1]), None, deltas[layer+1], wb[layer+1], activations[layer+1],
                     deltas[layer], np.int32(sizes[layer+1]), np.int32(sizes[layer+2]))
        for layer in range(3):
            update(queue, (weights[layer].size,), None, activations[layer], deltas[layer], wb[layer], mb[layer], vb[layer],
                   np.int32(sizes[layer]), np.int32(n), np.float32(1-0.9**step), np.float32(1-0.999**step))
    for destination, source in zip(weights, wb):
        cl.enqueue_copy(queue, destination, source)
    queue.finish()
    if not all(np.isfinite(w).all() for w in weights):
        raise RuntimeError("GPU training produced non-finite weights.")
    return [(w[:, :-1].copy(), w[:, -1].copy()) for w in weights], device.name
