"""
Build da extensao vip9000 (Cython) contra o runtime VIPLite do NPU VIP9000.

Uso (rodar NA PLACA/toolchain-alvo, ex.: aarch64 Radxa Cubie, dentro do
ambiente cross ou nativo com as libs do seu SDK):

    # NPU_SW_VERSION=1  -> viplite-tina v1.13 (libVIPlite.so + libVIPuser.so)
    # NPU_SW_VERSION=2  -> viplite-tina v2.0  (libNBGlinker.so + libVIPhal.so)
    NPU_SW_VERSION=2 python3 setup.py build_ext --inplace

Isso espelha exatamente a convencao usada pelos Makefiles da SDK original
(ai-sdk/examples/*/Makefile), que fazem:

    ifeq ("$(NPU_SW_VERSION)", "v2.0")
      INCLUDE  += -I.../v2.0/inc
      LIBS_VIP += -L.../v2.0/ -lNBGlinker -lVIPhal
      MYCFLAGS += -DNPU_SW_VERSION=2
    else
      INCLUDE  += -I.../v1.13/inc
      LIBS_VIP += -L.../v1.13/ -lVIPlite -lVIPuser
      MYCFLAGS += -DNPU_SW_VERSION=1
    endif

O .pxd/.pyx usam apenas a API que existe identicamente nas duas versoes
(mesmos nomes de funcao/enum em vip_lite.h v1.13 e v2.0 — validado por
compilacao cruzada contra os dois headers), entao o mesmo codigo Cython
serve para as duas, só troca o header/lib de destino.

Variaveis de ambiente aceitas:
  NPU_SW_VERSION  "1" ou "2" (default: "1")
  VIP_SDK_DIR     caminho para a pasta que contem inc/ e os .so
                  (default: ./viplite-tina/lib/aarch64-none-linux-gnu/v<versao>)
"""

import os
from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    raise SystemExit("Instale o Cython antes: pip install cython --break-system-packages")

NPU_SW_VERSION = os.environ.get("NPU_SW_VERSION", "1").strip()
if NPU_SW_VERSION not in ("1", "2"):
    raise SystemExit(f"NPU_SW_VERSION invalido: {NPU_SW_VERSION!r} (use '1' ou '2')")

_SDK_SUBDIR = {"1": "v1.13", "2": "v2.0"}[NPU_SW_VERSION]
_LIBRARIES = {"1": ["VIPlite", "VIPuser"], "2": ["NBGlinker", "VIPhal"]}[NPU_SW_VERSION]

VIP_SDK_DIR = os.environ.get(
    "VIP_SDK_DIR",
    f"./viplite-tina/lib/aarch64-none-linux-gnu/{_SDK_SUBDIR}",
)

include_dir = os.path.join(VIP_SDK_DIR, "inc")
lib_dir = VIP_SDK_DIR

extensions = [
    Extension(
        name="vip9000",
        sources=["vip9000.pyx"],
        include_dirs=[include_dir, np.get_include()],
        library_dirs=[lib_dir],
        libraries=_LIBRARIES,
        runtime_library_dirs=[lib_dir],  # embute rpath (Linux) p/ achar o .so em runtime
        extra_link_args=["-Wl,--no-as-needed", "-Wl,-rpath," + lib_dir, "-Wl,--disable-new-dtags"],
        # -DNPU_SW_VERSION=1|2, mesma convencao dos Makefiles originais da SDK.
        # O wrapper atual nao precisa disso (API igual nas duas versoes), mas
        # fica disponivel em C caso voce estenda vip9000.pyx com trechos
        # "#if NPU_SW_VERSION >= 2" (como faz examples/vpm_run/vpm_run.c).
        define_macros=[("NPU_SW_VERSION", NPU_SW_VERSION)],
        extra_compile_args=[f"-DVIP9000_NPU_SW_VERSION_STR={NPU_SW_VERSION}"],
    )
]

print(f"[vip9000/setup.py] NPU_SW_VERSION={NPU_SW_VERSION} "
      f"-> include={include_dir} libs={_LIBRARIES} lib_dir={lib_dir}")

setup(
    name="vip9000",
    version="0.2.0",
    description="Interface Cython para o NPU VeriSilicon/Vivante VIP9000 (VIPLite driver)",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        annotate=False,
        compile_time_env={"NPU_SW_VERSION": int(NPU_SW_VERSION)},
    ),
)
