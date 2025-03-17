from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='gaussianpro_ssim',
    ext_modules=[
        CUDAExtension('gaussianpro_ssim', 
            sources=[
                'PatchMatch.cpp', 
                'Propagation.cu',
                'pro.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
