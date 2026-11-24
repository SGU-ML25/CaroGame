"""Build lõi Cython tuỳ chọn cho AlphaBeta (L3) và PUCT (L7/L8).

    uv pip install cython setuptools        # hoặc: uv sync --extra native
    uv run python setup_native.py build_ext --inplace

Cần gcc hoặc clang (dùng `unsigned __int128`) trên Linux / macOS. Không build cũng không sao:
backend="auto" tự quay về bản Python nếu chưa build.
"""

from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    name="caro-native",
    ext_modules=cythonize(
        [
            Extension(
                "caro.engine._native",
                ["caro/engine/_native.pyx"],
                language="c++",
                extra_compile_args=["-O3", "-std=c++17"],
            ),
            Extension(
                "caro.engine._puct_native",
                ["caro/engine/_puct_native.pyx"],
                language="c++",
                extra_compile_args=["-O3", "-std=c++17", "-ffp-contract=off"],
            ),
        ],
        compiler_directives={"language_level": "3"},
    ),
    packages=[],
)
