#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "pyutils/pyutils.cuh"
#include "kittens.cuh"
using namespace kittens;

#define NUM_THREADS (kittens::WARP_THREADS)

#define _row 16
#define _col 32
struct fr_globals {
    using _gl = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
    _gl x;
    float accum;
    dim3 grid() { return dim3(_gl.batch, _gl.depth, _gl.rows);}
    dim3 block() { return dim3(_gl.cols);}
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void fr_tk(const __grid_constant__ fr_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int *)&__shm[0]);
    st_fl<_row, _col> (&x_s) = al.allocate<st_fl<_row, _col>>();

    rt_fl<_row, _col> x_reg;

    rt_fl<1, 1> accum_reg;
    zero(accum_reg);



    load(x_s, g.x, {0, 0, 0, 0});
    __syncthreads();

    load(x_reg, x_s);
    __syncthreads();

    mul(x_reg, x_reg, x_reg);
    __syncthreads();

    sum(accum_reg, x_reg);
    __syncthreads();

    load(g.accum, accum_reg);
}

void dispatch_fr_tk(fr_globals g) {
    fr_tk<<<g.grid(), g.block()>>>(g);
}

PYBIND11_MODULE(fr_norm, m) {
    m.doc() = "Frobenius norm";
    py::bind_kernel<fr_tk>(m, "fr_tk", &fr_globals::x, &fr_globals::accum);
    py::bind_function<dispatch_fr_tk>(m, "dispatch_fr_tk", &fr_globals);
    
}