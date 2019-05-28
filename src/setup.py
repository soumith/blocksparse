from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    'cxx': ['-O0'],
    'nvcc': ['-gencode=arch=compute_60,code=sm_60', '-DGOOGLE_CUDA=1'],
}

setup(
    name='blocksparse_cuda',    
    ext_modules=[
        CUDAExtension('blocksparse_cuda', [
            'transformer_op_gpu.cu',
            'transformer_op_pytorch.cu',
        ], extra_compile_args = extra_compile_args),
        
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
