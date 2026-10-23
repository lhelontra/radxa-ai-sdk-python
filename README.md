# vip9000 — Cython interface for the VIP9000 NPU (VIPLite)

Python/Cython binding for VeriSilicon/Vivante's **VIPLite** driver, which
controls the **VIP9000** NPU found on Allwinner SoCs (Radxa Cubie A733,
AI985, etc.). It supports both runtime versions distinguished by the SDK via
`NPU_SW_VERSION` (`1` = v1.13, `2` = v2.0 — see the Build section). The
interface was written and validated against the real API of:

```
ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v1.13/inc/vip_lite.h
ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v1.13/libVIPlite.so
```

(repository `ZIFENG278/ai-sdk`). That's the version whose symbols
(`vip_init`, `vip_create_network`, `vip_run_network`, etc.) are actually
exported by `libVIPlite.so` (confirmed via `nm -D`). The v2.0 build of the
same SDK ships an almost identical header, but in that folder the
implementation (`libVIPhal.so`) only exposes internal `viphal_*` symbols,
not the public `vip_*` API — so it isn't directly linkable without another
toolchain piece that isn't included in the repo (see below: it's actually
`libNBGlinker.so` that exports the public API for v2.0).

## Files

- `vip_lite.pxd` — `cdef extern` declarations mirroring `vip_lite.h`.
- `vip9000.pyx` — high-level Python/Cython wrapper (`vip9000.Network`, etc).
- `setup.py` — builds the extension, linking against the correct driver libs.
- `example_run.py` — example that loads a `.nb` (NBG) and runs inference.

## Build

This must be compiled **on the board** (or in an aarch64 cross
sysroot/toolchain), because the libraries (`libVIPlite.so`,
`libNBGlinker.so`, etc.) are proprietary ARM binaries.

The project supports both runtime versions that the SDK itself
distinguishes via `NPU_SW_VERSION` (same convention as the `Makefile`s in
`ai-sdk/examples/*`):

| `NPU_SW_VERSION` | SDK folder | Linked libs             | Example boards (`machinfo/*/config.mk`) |
|------------------|------------|--------------------------|-------------------------------------------|
| `1` (default)    | `v1.13`    | `-lVIPlite -lVIPuser`    | `ai985`, `t527`, `mr527`                  |
| `2`              | `v2.0`     | `-lNBGlinker -lVIPhal`   | `a733`, `mr536`, `t536`, `t736`           |

The public API (`vip_init`, `vip_create_network`, `vip_run_network`,
buffers, etc.) is identical between the two versions — I validated this by
compiling the Cython-generated `.c` file against both real headers
(`vip_lite.h` from v1.13 and from v2.0) with zero errors. The only thing
that changes is which `.so` actually exports those symbols: in v1.13 it's
`libVIPlite.so`; in v2.0, somewhat unintuitively, it's `libNBGlinker.so`
(which itself depends on `libVIPhal.so`) — `libVIPhal.so` on its own only
exposes internal `viphal_*` symbols.

```bash
pip install cython numpy --break-system-packages

# pick the version for your SoC/SDK (check machinfo/<board>/config.mk)
export NPU_SW_VERSION=2   # or 1

# point to the right folder (inc/ + .so). If omitted, setup.py assumes
# ./viplite-tina/lib/aarch64-none-linux-gnu/v2.0 (or v1.13) relative to cwd.
export VIP_SDK_DIR=/path/to/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0

python3 setup.py build_ext --inplace
```

After importing, `vip9000.BUILT_NPU_SW_VERSION` tells you which version was
actually linked into that build (`"1.13"` or `"2.0"`) — handy for
logging/debugging when you ship the compiled `.so` to different boards.

## Usage

```python
import vip9000
import numpy as np

vip9000.init()

# from a file path (VIP_CREATE_NETWORK_FROM_FILE)
with vip9000.Network("model.nb") as net:
    ...

# from an in-memory buffer (VIP_CREATE_NETWORK_FROM_MEMORY) — bytes,
# bytearray, memoryview, np.ndarray, or anything supporting the buffer
# protocol. Handy when the .nb comes from a download, an asset bundled in
# the app, a zip file, etc., without having to write it to disk first.
with open("model.nb", "rb") as f:
    nbg_bytes = f.read()

with vip9000.Network(nbg_bytes) as net:
    print(net.input_info())   # shape, dtype, quantization for each input
    print(net.output_info())
    outputs = net.run([np.zeros((1, 224, 224, 3), dtype=np.uint8)])

vip9000.destroy()
```

`vip9000.Network(source)` automatically picks between the two modes:
`str`/`os.PathLike` -> loads from a file; anything else (bytes, bytearray,
memoryview, np.ndarray, etc.) -> copies it into an internal `bytes` object
and loads it from memory via
`vip_create_network(..., VIP_CREATE_NETWORK_FROM_MEMORY, ...)`. That copy
is kept alive (`self._nbg_data`) for as long as the network object exists,
and released in `close()` — the driver header doesn't guarantee that
VIPLite copies 100% of the NBG content internally for every network type,
so out of caution we don't let the Python buffer get garbage-collected
early.

`Network.run()` copies the numpy arrays into the NPU's buffers
(`vip_map_buffer` + `memcpy` + `vip_flush_buffer`), triggers inference with
`vip_run_network` (synchronous), and reads the output back
(`vip_flush_buffer(INVALIDATE)` + `vip_map_buffer`), returning `np.ndarray`s
in the tensor's native dtype (without dequantizing — the
`tf_scale`/`tf_zero_point`/`quant_format` fields returned by
`input_info()`/`output_info()` give you what you need to dequantize
yourself if you want to).

## Limitations of this first version

- Only covers the most common path: `vip_create_network` from a `.nb` file
  or in-memory buffer (NBG generated by the Acuity SDK), synchronous
  execution (`vip_run_network`).
- Not exposed yet: `vip_group`/`vip_trigger_*` (async/batch execution),
  `vip_dup_network` (sharing weights between networks), buffers from
  `dma_buf`/physical address/handle, or power management
  (`vip_power_management`). These can be added following the same pattern
  as the `.pxd` — the header already declares all of it, only the Python
  wrappers are missing.
- Validated by type-checking against the real header (`gcc -c`), but not
  tested on actual hardware (I don't have access to a VIP9000 board in this
  environment) — I'd recommend running `example_run.py` with a small `.nb`
  (e.g. `examples/lenet/model/v2/lenet.nb` from `ai-sdk` itself) before
  going to production.
