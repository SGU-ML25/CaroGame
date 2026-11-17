"""Build lõi native tuỳ chọn (Cython) của cấp 3.

    uv pip install cython setuptools        # hoặc: uv sync --extra native
    uv run python setup_native.py build_ext --inplace

Cần gcc hoặc clang (dùng `unsigned __int128`) trên Linux / macOS. Không build cũng không sao:
AlphaBetaAgent(backend="python") vẫn chạy như cũ, backend="auto" tự quay về bản Python.
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
            )
        ],
        compiler_directives={"language_level": "3"},
    ),
    packages=[],
)
