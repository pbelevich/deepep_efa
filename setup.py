import os
import subprocess
import setuptools

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if __name__ == '__main__':
    # DeepEP-EFA: Always disable NVSHMEM, always enable EFA
    disable_nvshmem = True

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']

    # Core sources (same as DISABLE_NVSHMEM mode)
    sources = [
        'csrc/deep_ep.cpp',
        'csrc/kernels/runtime.cu',
        'csrc/kernels/layout.cu',
        'csrc/kernels/intranode.cu',
    ]

    # EFA transport sources (C++ only, compiled by host compiler)
    efa_sources = [
        'csrc/transport/efa_transport.cpp',
        'csrc/transport/gdr_signal.cpp',
        'csrc/transport/imm_counter.cpp',
        'csrc/transport/efa_worker.cpp',
        'csrc/transport/ll_pipeline.cpp',
    ]
    sources.extend(efa_sources)

    # EFA internode kernel (CUDA)
    sources.append('csrc/kernels/internode_efa.cu')

    # EFA fused kernels (CUDA) — moe_routing_sort, topk_remap, efa_permute, build_recv_src_meta
    sources.append('csrc/kernels/efa_kernels.cu')

    # Cooperative kernels (CUDA) — coop dispatch/combine with GDRCopy signaling
    coop_sources = [
        'csrc/kernels/coop_dispatch_send.cu',
        'csrc/kernels/coop_dispatch_recv.cu',
        'csrc/kernels/coop_combine_send.cu',
        'csrc/kernels/coop_combine_recv.cu',
    ]
    sources.extend(coop_sources)

    include_dirs = ['csrc/']
    library_dirs = []
    extra_link_args = ['-lcuda']

    # Always set DISABLE_NVSHMEM to suppress NVSHMEM code paths
    cxx_flags.append('-DDISABLE_NVSHMEM')
    nvcc_flags.append('-DDISABLE_NVSHMEM')

    # Enable EFA mode
    cxx_flags.append('-DENABLE_EFA')
    nvcc_flags.append('-DENABLE_EFA')

    # libfabric flags
    cxx_flags.extend(['-I/opt/amazon/efa/include'])
    extra_link_args.extend(['-L/opt/amazon/efa/lib', '-lfabric'])

    # GDRCopy flags
    cxx_flags.extend(['-I/usr/include'])
    extra_link_args.extend(['-lgdrapi'])

    # pthread for worker threads
    extra_link_args.append('-lpthread')

    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')
    else:
        # Prefer H800 series
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

        # CUDA 12 flags
        # NOTE: We do NOT use -rdc=true since we don't have NVSHMEM device symbols
        nvcc_flags.extend(['--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }

    # Summary
    print('Build summary (DeepEP-EFA):')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > NVSHMEM: disabled')
    print(f' > EFA: enabled')
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(name='deep_ep',
                     version='2.0.0a0' + revision,
                     packages=setuptools.find_packages(include=['deep_ep']),
                     ext_modules=[
                         CUDAExtension(name='deep_ep_cpp',
                                       include_dirs=include_dirs,
                                       library_dirs=library_dirs,
                                       sources=sources,
                                       extra_compile_args=extra_compile_args,
                                       extra_link_args=extra_link_args)
                     ],
                     cmdclass={'build_ext': BuildExtension})
