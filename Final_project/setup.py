"""Build the _symnmf C extension so Python can import it."""
from setuptools import setup, Extension

symnmf_ext = Extension(
    '_symnmf',
    sources=['symnmfmodule.c', 'symnmf.c', 'matrix_ops.c'],
    extra_compile_args=['-std=c99', '-Wall', '-Wextra', '-Werror', '-pedantic-errors'],
)

setup(
    name='symnmf',
    version='1.0',
    ext_modules=[symnmf_ext],
)
