# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

import os
import re
import sys
import subprocess
import shutil
import glob
import importlib.util

from setuptools import setup, find_packages

# ============================================================
# Parse build mode
# ============================================================
FLASHMASK_BUILD = os.environ.get('FLASHMASK_BUILD', 'all').lower()
requested_components = set(re.split(r'[,\s+]+', FLASHMASK_BUILD.strip()))
requested_components.discard('')
ALLOWED_COMPONENTS = {'core', 'cpb', 'utils', 'ovl', 'all'}
invalid_components = requested_components - ALLOWED_COMPONENTS
assert requested_components and not invalid_components, (
    f"Invalid FLASHMASK_BUILD component(s): {', '.join(sorted(invalid_components or requested_components))}. "
    f"Allowed: {', '.join(sorted(ALLOWED_COMPONENTS))}. "
    f"Combinations e.g. 'core+cpb', 'core+ovl'."
)

_build_all = 'all' in requested_components
# The CuTe DSL kernels are the whole point of the package: always shipped.
BUILD_CORE = True
BUILD_CPB = _build_all or 'cpb' in requested_components
BUILD_UTILS = _build_all or 'utils' in requested_components

# Overlap bridge needs NCCL GIN + H/B (sm_90a/sm_100). Built as part of 'all',
# or on its own (e.g. FLASHMASK_BUILD=core+ovl).
BUILD_OVL = _build_all or 'ovl' in requested_components

# ============================================================
# Parse target framework
# ============================================================
# The CuTe DSL core serves paddle and torch from one implementation (see
# flashmask/backend), so this switch controls only one thing: whether the
# paddle-only CUDA submodule is built. 'cpb' is now framework-neutral (raw
# pointer-ABI .so driven via flashmask.backend), so it builds for torch too.
# 'utils' is still a Paddle custom op (paddle::Tensor + PD_BUILD_OP) and cannot
# be built for torch yet; it is skipped rather than failing the whole install
# (with a warning only when 'utils' was requested explicitly, not via 'all').
# The framework itself is NOT added as a dependency here -- see the note by
# install_requires for why.
FLASHMASK_FRAMEWORK = os.environ.get('FLASHMASK_FRAMEWORK', 'paddle').lower().strip()
ALLOWED_FRAMEWORKS = {'paddle', 'torch', 'both'}
assert FLASHMASK_FRAMEWORK in ALLOWED_FRAMEWORKS, (
    f"Invalid FLASHMASK_FRAMEWORK={FLASHMASK_FRAMEWORK!r}. "
    f"Allowed: {', '.join(sorted(ALLOWED_FRAMEWORKS))} (default: paddle)."
)
WANT_PADDLE = FLASHMASK_FRAMEWORK in ('paddle', 'both')

if not WANT_PADDLE and BUILD_UTILS:
    if 'utils' in requested_components:
        print("[flashmask] WARNING: 'utils' is a Paddle custom op and cannot "
              f"be built for FLASHMASK_FRAMEWORK={FLASHMASK_FRAMEWORK}; skipping it. "
              "The CuTe DSL core and 'cpb' are unaffected.")
    BUILD_UTILS = False

print(f"[flashmask] FLASHMASK_BUILD={FLASHMASK_BUILD}  "
      f"FLASHMASK_FRAMEWORK={FLASHMASK_FRAMEWORK}  "
      f"BUILD_CORE={BUILD_CORE}  BUILD_UTILS={BUILD_UTILS}  "
      f"BUILD_CPB={BUILD_CPB}  BUILD_OVL={BUILD_OVL}")

# ============================================================
# Config
# ============================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FLASHMASK_DIR = os.path.join(ROOT_DIR, 'flashmask')

_BASE_VERSION = '4.0.0'

# ============================================================
# Version: _BASE_VERSION + git commit hash
# ============================================================
def _get_version():
    """Build PEP 440 version: _BASE_VERSION+gCOMMIT"""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=ROOT_DIR, stderr=subprocess.DEVNULL,
        ).decode('ascii').strip()
    except Exception:
        return _BASE_VERSION
    return f"{_BASE_VERSION}+g{commit}"

VERSION = _get_version()

# ============================================================
# Packages: exclude modules not being built
# ============================================================
exclude_packages = ['build', 'build.*', 'tests', 'tests.*',
                     'benchmarks', 'benchmarks.*',
                     'flashmask.cp_balance.csrc', 'flashmask.cp_balance.csrc.*',
                     'flashmask.utils.csrc', 'flashmask.utils.csrc.*',
                     'flashmask.overlap.csrc', 'flashmask.overlap.csrc.*']
if not BUILD_CPB:
    exclude_packages += [
        "flashmask.cp_balance",
        "flashmask.cp_balance.*",
    ]
if not BUILD_OVL:
    exclude_packages += [
        "flashmask.overlap",
        "flashmask.overlap.*",
    ]

packages = find_packages(exclude=exclude_packages)

# ============================================================
# Dependencies
# ============================================================
install_requires = [
    'typing_extensions',
    'nvidia-cutlass==4.2.0.0',
    'nvidia-cutlass-dsl[cu13]==4.5.2',
    "apache-tvm-ffi>=0.1.5,<0.2.0",
]

# Neither framework is declared as a dependency, on purpose. In a training
# image torch/paddle are pre-installed as a specific CUDA build; declaring them
# buys nothing (they are already importable) and forces pip to re-resolve the
# framework's whole dependency closure against the configured index. On an
# internal index that lacks, e.g., nvidia-nccl-cu13, that resolution fails even
# though the framework itself is present. FLASHMASK_FRAMEWORK therefore only
# selects which C++ submodules to build; the framework is supplied by the
# environment. Both frontends use only long-stable framework APIs.

# `pip install flashmask[dev]` for the test tooling. No torch/paddle extra, for
# the reason above.
extras_require = {
    'dev': ['pytest'],
}

# ============================================================
# Pre-install dependencies
# (python setup.py install / pip install --no-build-isolation
#  won't auto-install install_requires before running setup.py,
#  so we do it explicitly here.)
# ============================================================
def _ensure_deps(deps):
    """pip install missing dependencies before build."""
    # pip package name -> actual import name. These are the ones where the
    # distribution name does not simply map to the module name:
    #   nvidia-cutlass-dsl has no top-level module of its own -- it installs
    #   into the `cutlass` namespace, and `cutlass.base_dsl` is the piece that
    #   only exists once the DSL package is present.
    #   apache-tvm-ffi installs `tvm_ffi`, not `tvm.ffi`.
    _IMPORT_MAP = {
        'nvidia-cutlass': 'cutlass',
        'nvidia-cutlass-dsl': 'cutlass.base_dsl',
        'apache-tvm-ffi': 'tvm_ffi',
        'typing_extensions': 'typing_extensions',
    }
    missing = []
    for dep in deps:
        # strip version specifiers and extras
        pkg_name = dep.split('[')[0]  # remove extras like [cu13]
        pkg_name = pkg_name.split('==')[0].split('>=')[0].split('<=')[0].strip()
        import_name = _IMPORT_MAP.get(pkg_name, pkg_name.replace('-', '_'))
        try:
            __import__(import_name)
        except ImportError:
            missing.append(dep)
    if missing:
        print(f"[flashmask] Installing missing dependencies: {missing}")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install'] + missing,
            )
        except subprocess.CalledProcessError as e:
            print(f"[flashmask] WARNING: pip install failed ({e}), continuing...")

_ensure_deps(install_requires)

# ============================================================
# CUDA submodule builder
# ============================================================
# The optional CUDA submodules need different nvcc flags from each other
# (e.g. cp_balance targets sm_80/sm_90a/sm_100), so each one is compiled
# independently through its own setup.py. This function handles:
#   1. Run the submodule's own setup.py build_ext
#   2. Copy the resulting .so into the Python package directory
#   3. Return the package name for package_data (so the .so ships in the wheel)
#
# To add a new submodule, just call _build_cuda_submodule() and append
# the returned package name to _submodule_package_data.

def _detect_cutlass_inc():
    """Find a cutlass include dir containing cutlass/bfloat16.h.

    Priority:
      1. env FM4_OVERLAP_CUTLASS_INC (explicit override)
      2. the pip-installed nvidia-cutlass package (cutlass_library/source/include,
         or cutlass/include depending on the wheel layout)
      3. Paddle's vendored cutlass (third_party/cutlass/include) if discoverable
    Returns the dir, or None if nothing usable is found.
    """
    env = os.environ.get('FM4_OVERLAP_CUTLASS_INC')
    candidates = []
    if env:
        candidates.append(env)
    # pip install nvidia-cutlass ships the C++ headers under cutlass_library.
    try:
        import cutlass_library  # noqa: F401
        _cll = os.path.dirname(os.path.abspath(cutlass_library.__file__))
        candidates.append(os.path.join(_cll, 'source', 'include'))
    except Exception:
        pass
    try:
        import cutlass  # noqa: F401
        _cl = os.path.dirname(os.path.abspath(cutlass.__file__))
        candidates.append(os.path.join(_cl, 'include'))
    except Exception:
        pass
    # Paddle install layout: <paddle>/third_party/cutlass/include. Try to locate
    # paddle and derive it, best-effort (skipped silently if paddle absent).
    try:
        import paddle  # noqa: F401
        _pd = os.path.dirname(os.path.abspath(paddle.__file__))
        # paddle/__init__.py -> .../site-packages/paddle ; cutlass usually lives
        # in the source tree, so also probe a couple of common roots.
        for _root in (
            os.path.join(_pd, '..', '..', '..', 'third_party', 'cutlass', 'include'),
        ):
            candidates.append(os.path.normpath(_root))
    except Exception:
        pass

    for c in candidates:
        if c and os.path.exists(os.path.join(c, 'cutlass', 'bfloat16.h')):
            return c
    return None


def _detect_nccl_home():
    """NCCL install dir: NCCL_HOME/NCCL_ROOT, else this interpreter's wheel.

    Preferring the interpreter's own wheel is what keeps build and runtime from
    drifting: overlap_runtime.py pins the same file.
    """
    env = os.environ.get('NCCL_HOME') or os.environ.get('NCCL_ROOT')
    if env:
        return env
    spec = importlib.util.find_spec('nvidia.nccl')
    if spec is None or not spec.submodule_search_locations:
        return None
    return next(iter(spec.submodule_search_locations))


def _detect_nccl_version(nccl_home):
    """Read NCCL_MAJOR/MINOR/PATCH from the selected NCCL installation."""
    header = os.path.join(nccl_home, 'include', 'nccl.h')
    try:
        with open(header, encoding='utf-8') as f:
            text = f.read()
    except (OSError, UnicodeError):
        return None

    version = []
    for name in ('MAJOR', 'MINOR', 'PATCH'):
        match = re.search(
            rf'^\s*#\s*define\s+NCCL_{name}\s+(\d+)\b',
            text,
            re.MULTILINE,
        )
        if match is None:
            return None
        version.append(int(match.group(1)))
    return tuple(version)


def _detect_cuda_arch():
    """';'-joined sm arch list for -gencode: FM4_OVERLAP_CUDA_ARCH, else the local GPU.

    The env var takes one arch or a ','/';' list, e.g. '100a;103a'.
    """
    env = os.environ.get('FM4_OVERLAP_CUDA_ARCH')
    if env:
        archs = [a for a in re.split(r'[,;\s]+', env.strip()) if a]
        bad = [a for a in archs if not re.fullmatch(r'\d+a?', a)]
        assert archs and not bad, (
            f"Invalid FM4_OVERLAP_CUDA_ARCH={env!r}: expected sm archs like "
            f"'100a' or '100a;103a', got {bad or archs}"
        )
        return ';'.join(archs)
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            text=True, stderr=subprocess.DEVNULL)
        major, minor = out.strip().splitlines()[0].strip().split('.')
        arch = f'{int(major)}{int(minor)}'
        # The accelerated 'a' suffix only exists from sm_90 on.
        return arch + 'a' if int(major) >= 9 else arch
    except Exception:
        return '90a'


def _build_cmake_submodule(name, csrc_dir, pkg_dir, lib_prefix, cmake_defs):
    """Build a submodule via a standalone sub-CMake (configure + build).

    Unlike _build_cuda_submodule (Paddle CUDAExtension), this drives plain CMake.
    Used by the standalone NCCL GIN overlap bridge. Steps:
      1. cmake -S csrc_dir -B csrc_dir/build  <-D defs...>
      2. cmake --build csrc_dir/build -j
      3. copy the produced lib{lib_prefix}*.so into pkg_dir
      4. clean the build dir

    Args:
        name: human-readable label for logs.
        csrc_dir: dir containing the sub-CMakeLists.txt.
        pkg_dir: package dir to copy the .so into.
        lib_prefix: shared-lib basename prefix to glob (e.g. 'fm4_overlap').
        cmake_defs: dict of -D cache vars.

    Returns the dotted package name for package_data, or None if skipped.
    """
    if not os.path.isdir(csrc_dir):
        print(f"[flashmask] {name}: csrc directory not found, skipping.")
        return None

    build_dir = os.path.join(csrc_dir, 'build')
    print(f"[flashmask] Building {name} via CMake...")

    configure = ['cmake', '-S', csrc_dir, '-B', build_dir,
                 '-DCMAKE_BUILD_TYPE=Release']
    for k, v in cmake_defs.items():
        configure.append(f'-D{k}={v}')

    result = subprocess.run(configure, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[flashmask] {name} cmake configure STDOUT:\n{result.stdout}")
        print(f"[flashmask] {name} cmake configure STDERR:\n{result.stderr}")
        raise RuntimeError(
            f"Failed to configure {name}.\n"
            f"Run manually: {' '.join(configure)}"
        )

    # Stream output live (no capture) + --verbose so every nvcc command line
    # and per-TU progress is visible; bound -j to avoid oversubscribing nvcc
    # on the heavy sm_103a compiles.
    result = subprocess.run(
        ['cmake', '--build', build_dir, '-j', '8', '--verbose'],
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build {name} (see streamed build output above).\n"
            f"Run manually: cmake --build {build_dir} -j 8 --verbose"
        )

    so_files = glob.glob(os.path.join(build_dir, '**', f'lib{lib_prefix}*.so'),
                         recursive=True)
    if not so_files:
        raise RuntimeError(
            f"{name} build succeeded but no lib{lib_prefix}*.so found under "
            f"{build_dir}/"
        )
    so_path = so_files[0]
    shutil.copy2(so_path, os.path.join(pkg_dir, os.path.basename(so_path)))
    print(f"[flashmask] {name} built: {os.path.basename(so_path)}")

    shutil.rmtree(build_dir, ignore_errors=True)

    return os.path.relpath(pkg_dir, ROOT_DIR).replace(os.sep, '.')


def _build_cuda_submodule(name, csrc_dir, pkg_dir):
    """Build a CUDA submodule and copy outputs into its package directory.

    Paddle's build_ext produces in build/:
      - {module_name}.so   — compiled CUDA binary (no _pd_ suffix)
      - {module_name}.py   — Python wrapper that loads {module_name}_pd_.so
    The wrapper hardcodes the _pd_ filename, so we rename the .so when copying.

    Args:
        name: Human-readable name for log messages.
        csrc_dir: Directory containing the submodule's setup.py.
        pkg_dir: Python package directory to copy outputs into.

    Returns:
        Package name (dot-separated) for package_data, or None if skipped.
    """
    if not os.path.isdir(csrc_dir):
        print(f"[flashmask] {name}: csrc directory not found, skipping.")
        return None

    print(f"[flashmask] Building {name} CUDA extension...")
    result = subprocess.run(
        [sys.executable, 'setup.py', 'build_ext'],
        cwd=csrc_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[flashmask] {name} build STDERR:\n{result.stderr}")
        raise RuntimeError(
            f"Failed to build {name} CUDA extension.\n"
            f"Build manually: cd {csrc_dir} && python setup.py build_ext"
        )

    # Find the .so and wrapper .py in build/
    so_files = glob.glob(os.path.join(csrc_dir, 'build', '**', '*.so'), recursive=True)
    if not so_files:
        raise RuntimeError(
            f"{name} build_ext succeeded but no .so found under "
            f"{os.path.join(csrc_dir, 'build')}/"
        )
    so_path = so_files[0]
    module_name = os.path.basename(so_path).replace('.so', '')
    wrapper_path = os.path.join(os.path.dirname(so_path), f'{module_name}.py')
    if not os.path.exists(wrapper_path):
        raise RuntimeError(
            f"{name}: Paddle-generated wrapper {module_name}.py not found "
            f"alongside {so_path}"
        )

    # Copy to pkg_dir. Rename .so to add _pd_ suffix (wrapper hardcodes this name).
    shutil.copy2(so_path, os.path.join(pkg_dir, f'{module_name}_pd_.so'))
    shutil.copy2(wrapper_path, pkg_dir)
    print(f"[flashmask] {name} built: {module_name}_pd_.so + {module_name}.py")

    # Clean up build artifacts from csrc_dir
    for _d in glob.glob(os.path.join(csrc_dir, 'build')) + \
              glob.glob(os.path.join(csrc_dir, '*.egg-info')):
        shutil.rmtree(_d, ignore_errors=True)
    # Also clean any _pd_.so / wrapper .py that Paddle may leave in csrc_dir
    for _f in glob.glob(os.path.join(csrc_dir, '*_pd_.so')) + \
              glob.glob(os.path.join(csrc_dir, f'{module_name}.py')):
        os.remove(_f)

    # Derive package name from pkg_dir relative to ROOT_DIR
    # e.g. flashmask/cp_balance -> flashmask.cp_balance
    return os.path.relpath(pkg_dir, ROOT_DIR).replace(os.sep, '.')


# ============================================================
# Build CUDA submodules
# ============================================================
_submodule_package_data = {}

# --- cp_balance: framework-neutral pointer-ABI .so, built via standalone CMake ---
# The load-balancing kernels are driven from Python via ctypes + flashmask.backend,
# so cpb builds for paddle AND torch (targets the local GPU arch; sm_80+).
if BUILD_CPB:
    _pkg = _build_cmake_submodule(
        'CP Balance',
        csrc_dir=os.path.join(FLASHMASK_DIR, 'cp_balance', 'csrc'),
        pkg_dir=os.path.join(FLASHMASK_DIR, 'cp_balance'),
        lib_prefix='flashmask_cpbalance',
        cmake_defs={'CPB_CUDA_ARCH': _detect_cuda_arch()},
    )
    if _pkg:
        _submodule_package_data[_pkg] = ['*.so']

# --- utils: small CUDA helpers used by the CuTe DSL Python paths ---
if BUILD_UTILS:
    _pkg = _build_cuda_submodule(
        'FlashMask utils',
        csrc_dir=os.path.join(FLASHMASK_DIR, 'utils', 'csrc'),
        pkg_dir=os.path.join(FLASHMASK_DIR, 'utils'),
    )
    if _pkg:
        _submodule_package_data[_pkg] = ['*.so']

# To add future submodules, just repeat:
#   _pkg = _build_cuda_submodule('Name', csrc_dir=..., pkg_dir=...)
#   if _pkg:
#       _submodule_package_data[_pkg] = ['*.so']

# --- overlap bridge: NCCL GIN + sm_90a/sm_100, built via standalone sub-CMake ---
# Built when 'all' or 'ovl' is requested (BUILD_OVL); targets H/B only. NCCL
# install and target arch are auto-detected; NCCL_HOME/NCCL_ROOT and
# FM4_OVERLAP_CUDA_ARCH override. Missing NCCL or cutlass fails the build under
# 'ovl'; under 'all' it only skips the bridge.
if BUILD_OVL:
    _ovl_csrc = os.path.join(FLASHMASK_DIR, 'overlap', 'csrc')
    _ovl_pkg_dir = os.path.join(FLASHMASK_DIR, 'overlap')
    _nccl_home = _detect_nccl_home()
    _nccl_version = (
        _detect_nccl_version(_nccl_home)
        if _nccl_home and os.path.isdir(_nccl_home)
        else None
    )
    _cutlass_inc = _detect_cutlass_inc()

    _ovl_missing = None
    if not _nccl_home or not os.path.isdir(_nccl_home):
        _ovl_missing = (f"no NCCL install found ({_nccl_home}). Install "
                        "nvidia-nccl-cu13 into this interpreter, or set NCCL_HOME")
    elif _cutlass_inc is None:
        _ovl_missing = ("cutlass/bfloat16.h not found (set FM4_OVERLAP_CUTLASS_INC "
                        "or install nvidia-cutlass)")
    elif _nccl_version is None:
        _ovl_missing = ("NCCL version not found in "
                        f"{os.path.join(_nccl_home, 'include', 'nccl.h')}")
    elif _nccl_version < (2, 30, 5):
        _ovl_missing = (f"NCCL {'.'.join(map(str, _nccl_version))} at {_nccl_home} "
                        "is too old; requires NCCL >= 2.30.5")

    if _ovl_missing is not None and 'ovl' in requested_components:
        # Fail now; otherwise every rank fails at its first overlap call.
        raise RuntimeError(f"[flashmask] FLASHMASK_BUILD={FLASHMASK_BUILD} requests "
                           f"'ovl', but the overlap bridge cannot be built: {_ovl_missing}.")
    if _ovl_missing is not None:
        print(f"[flashmask] overlap: {_ovl_missing}; skipping overlap bridge.")
    else:
        _ovl_arch = _detect_cuda_arch()
        print(f"[flashmask] overlap: NCCL_HOME={_nccl_home}  "
              f"arch={_ovl_arch}  cutlass_inc={_cutlass_inc}")
        _pkg = _build_cmake_submodule(
            'FM4 Overlap',
            csrc_dir=_ovl_csrc,
            pkg_dir=_ovl_pkg_dir,
            lib_prefix='fm4_overlap',
            cmake_defs={
                'NCCL_INSTALL_DIR': _nccl_home,
                'FM4_OVERLAP_CUDA_ARCH': _ovl_arch,
                'FM4_OVERLAP_CUTLASS_INC': _cutlass_inc,
            },
        )
        if _pkg:
            _submodule_package_data[_pkg] = ['*.so']

# ============================================================
# Build: pure Python package (the CuTe DSL core has nothing to compile)
# ============================================================
setup(
    name='flashmask',
    version=VERSION,
    packages=packages,
    package_data=_submodule_package_data,
    author='PaddlePaddle',
    description='FlashMask: Efficient and Rich Mask Extension of FlashAttention',
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires='>=3.10',
)
