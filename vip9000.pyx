# vip9000.pyx
# cython: language_level=3
"""
Interface Cython de alto nivel para o NPU VeriSilicon/Vivante VIP9000,
usando o driver de baixo nivel VIPLite (vip_lite.h / libVIPlite.so).

Fluxo tipico:

    import vip9000

    vip9000.init()                     # uma vez por processo
    net = vip9000.Network("model.nb")  # NBG gerado pelo Acuity SDK (arquivo)
    out = net.run([input_array])       # list[np.ndarray] -> list[np.ndarray]
    net.close()
    vip9000.destroy()

Tambem e possivel carregar o NBG a partir de um buffer ja em memoria
(bytes, bytearray, memoryview, np.ndarray, resultado de um download, um
recurso empacotado no binario, etc.), sem precisar gravar em disco:

    with open("model.nb", "rb") as f:
        nbg_bytes = f.read()

    net = vip9000.Network(nbg_bytes)   # detecta automaticamente que
                                        # nao e um path e usa
                                        # VIP_CREATE_NETWORK_FROM_MEMORY

Formatos de tensor suportados no mapeamento numpy: fp32, fp16 (via uint16
cru), uint8, int8, uint16, int16, int32, uint32.
"""

import os
import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t
from libc.string cimport memcpy

cimport vip_lite as c

cnp.import_array()

# NPU_SW_VERSION e injetado em tempo de compilacao pelo setup.py via
# cythonize(..., compile_time_env={"NPU_SW_VERSION": 1 ou 2}), espelhando
# o -DNPU_SW_VERSION=1|2 dos Makefiles originais da SDK (ex.:
# examples/vpm_run/vpm_run.c usa "#if NPU_SW_VERSION >= 2" para as poucas
# diferencas entre viplite-tina v1.13 e v2.0). A API que este wrapper usa
# (init/create_network/buffers/run) e identica nas duas versoes, entao nao
# precisamos de "#if" no meio do codigo — so registramos qual foi linkada.
IF NPU_SW_VERSION == 2:
    BUILT_NPU_SW_VERSION = "2.0"
ELSE:
    BUILT_NPU_SW_VERSION = "1.13"


class VipError(RuntimeError):
    """Erro retornado por uma chamada vip_* (VIP_ERROR_*)."""
    pass


cdef _check(c.vip_status_e status, str where):
    if status != c.VIP_SUCCESS:
        raise VipError(f"{where} falhou com status {status}")


_DTYPE_MAP = {
    c.VIP_BUFFER_FORMAT_FP32:  np.float32,
    c.VIP_BUFFER_FORMAT_UINT8: np.uint8,
    c.VIP_BUFFER_FORMAT_INT8:  np.int8,
    c.VIP_BUFFER_FORMAT_UINT16: np.uint16,
    c.VIP_BUFFER_FORMAT_INT16:  np.int16,
    c.VIP_BUFFER_FORMAT_INT32:  np.int32,
    c.VIP_BUFFER_FORMAT_UINT32: np.uint32,
}


def init():
    """Inicializa o hardware VIP9000 e o runtime VIPLite. Chamar 1x por processo."""
    _check(c.vip_init(), "vip_init")


def destroy():
    """Encerra o runtime VIPLite e desliga o VIP9000."""
    _check(c.vip_destroy(), "vip_destroy")


def get_version():
    """Retorna a versao (uint32) do driver VIPLite."""
    return c.vip_get_version()


def query_hardware():
    """Retorna um dict com informacoes basicas de hardware (cid, device_count, core_count)."""
    cdef c.vip_uint32_t cid = 0
    cdef c.vip_uint32_t device_count = 0
    cdef cnp.uint32_t[:] buf_view

    _check(c.vip_query_hardware(c.VIP_QUERY_HW_PROP_CID,
                                 sizeof(c.vip_uint32_t), &cid), "vip_query_hardware(CID)")
    _check(c.vip_query_hardware(c.VIP_QUERY_HW_PROP_DEVICE_COUNT,
                                 sizeof(c.vip_uint32_t), &device_count),
           "vip_query_hardware(DEVICE_COUNT)")

    cores = []
    if device_count > 0:
        buf = np.zeros(device_count, dtype=np.uint32)
        buf_view = buf
        _check(c.vip_query_hardware(c.VIP_QUERY_HW_PROP_CORE_COUNT_EACH_DEVICE,
                                     sizeof(c.vip_uint32_t) * device_count,
                                     &buf_view[0]),
               "vip_query_hardware(CORE_COUNT_EACH_DEVICE)")
        cores = buf.tolist()

    return {"cid": cid, "device_count": device_count, "core_count_each_device": cores}


cdef class _Buffer:
    """Wrapper interno de um vip_buffer (entrada ou saida de uma rede)."""
    cdef c.vip_buffer handle
    cdef public tuple shape
    cdef public object dtype
    cdef public str name
    cdef public double tf_scale
    cdef public int tf_zero_point
    cdef public int quant_format

    def __cinit__(self):
        self.handle = NULL

    cdef _destroy(self):
        if self.handle is not NULL:
            c.vip_destroy_buffer(self.handle)
            self.handle = NULL

    def __dealloc__(self):
        self._destroy()


ctypedef c.vip_status_e (*_query_func_t)(c.vip_network, c.vip_uint32_t,
                                          c.vip_enum, void *)


cdef _Buffer _make_buffer_for_index(c.vip_network network, c.vip_uint32_t index,
                                     bint is_input):
    """Consulta os metadados de um input/output pelo indice e cria o vip_buffer
    correspondente (o driver aloca a memoria; nos so a preenchemos/lemos)."""
    cdef c.vip_buffer_create_params_t params
    cdef c.vip_uint32_t num_dims = 0
    cdef c.vip_uint32_t data_format = 0
    cdef c.vip_uint32_t quant_format = 0
    cdef c.vip_uint32_t fixed_point_pos = 0
    cdef c.vip_float_t tf_scale = 0.0
    cdef c.vip_uint32_t tf_zero_point = 0
    cdef char name_buf[64]
    cdef _query_func_t query

    if is_input:
        query = c.vip_query_input
    else:
        query = c.vip_query_output

    _check(query(network, index, c.VIP_BUFFER_PROP_NUM_OF_DIMENSION, &num_dims),
           "vip_query_input/output(NUM_OF_DIMENSION)")
    _check(query(network, index, c.VIP_BUFFER_PROP_DATA_FORMAT, &data_format),
           "vip_query_input/output(DATA_FORMAT)")
    _check(query(network, index, c.VIP_BUFFER_PROP_QUANT_FORMAT, &quant_format),
           "vip_query_input/output(QUANT_FORMAT)")

    params.num_of_dims = num_dims
    params.data_format = data_format
    params.quant_format = quant_format
    params.memory_type = 0  # VIP_BUFFER_MEMORY_TYPE_DEFAULT

    cdef c.vip_uint32_t dims[6]
    _check(query(network, index, c.VIP_BUFFER_PROP_SIZES_OF_DIMENSION, &dims[0]),
           "vip_query_input/output(SIZES_OF_DIMENSION)")
    for i in range(num_dims):
        params.sizes[i] = dims[i]

    tf_scale_val = 0.0
    tf_zero_point_val = 0
    if quant_format == c.VIP_BUFFER_QUANTIZE_TF_ASYMM:
        query(network, index, c.VIP_BUFFER_PROP_TF_SCALE, &tf_scale)
        query(network, index, c.VIP_BUFFER_PROP_TF_ZERO_POINT, &tf_zero_point)
        tf_scale_val = tf_scale
        tf_zero_point_val = tf_zero_point

    query(network, index, c.VIP_BUFFER_PROP_NAME, name_buf)

    cdef c.vip_buffer handle = NULL
    _check(c.vip_create_buffer(&params, sizeof(params), &handle), "vip_create_buffer")

    buf = _Buffer()
    buf.handle = handle
    shape = tuple(int(dims[i]) for i in range(num_dims))
    buf.shape = shape
    buf.dtype = _DTYPE_MAP.get(data_format, np.uint8)
    buf.name = name_buf.decode("utf-8", "ignore")
    buf.tf_scale = tf_scale_val
    buf.tf_zero_point = tf_zero_point_val
    buf.quant_format = quant_format
    return buf


cdef class Network:
    """Uma rede neural carregada no VIP9000 a partir de um binario NBG (.nb)."""

    cdef c.vip_network handle
    cdef public list inputs
    cdef public list outputs
    cdef bint _prepared
    cdef bytes _nbg_data  # mantem o buffer do NBG vivo enquanto a rede existir

    def __cinit__(self):
        self.handle = NULL
        self.inputs = []
        self.outputs = []
        self._prepared = False
        self._nbg_data = None

    def __init__(self, source):
        """Cria e prepara a rede a partir de `source`, que pode ser:

        - um caminho de arquivo .nb/NBG (str ou os.PathLike) -> usa
          VIP_CREATE_NETWORK_FROM_FILE; ou
        - o conteudo do NBG ja em memoria (bytes, bytearray, memoryview,
          np.ndarray de bytes, ou qualquer objeto com buffer protocol) ->
          usa VIP_CREATE_NETWORK_FROM_MEMORY.

        Uma referencia ao buffer e mantida em self._nbg_data ate close(),
        por seguranca (o header nao garante que o driver copia 100% do
        conteudo internamente para todo tipo de NBG).
        """
        cdef c.vip_network net = NULL
        cdef bytes path_b
        cdef const char *path_c
        cdef const unsigned char[::1] view
        cdef c.vip_uint32_t size

        if isinstance(source, (str, os.PathLike)):
            path_b = os.fspath(source).encode("utf-8")
            path_c = path_b
            _check(c.vip_create_network(path_c, 0, c.VIP_CREATE_NETWORK_FROM_FILE, &net),
                   "vip_create_network (arquivo)")
        else:
            # buffer em memoria: bytes/bytearray/memoryview/np.ndarray/etc.
            self._nbg_data = bytes(source)
            size = <c.vip_uint32_t> len(self._nbg_data)
            if size == 0:
                raise ValueError("buffer NBG vazio")
            view = self._nbg_data
            _check(c.vip_create_network(&view[0], size,
                                         c.VIP_CREATE_NETWORK_FROM_MEMORY, &net),
                   "vip_create_network (memoria)")

        self.handle = net

        _check(c.vip_prepare_network(self.handle), "vip_prepare_network")
        self._prepared = True

        cdef c.vip_uint32_t input_count = 0
        cdef c.vip_uint32_t output_count = 0
        _check(c.vip_query_network(self.handle, c.VIP_NETWORK_PROP_INPUT_COUNT,
                                    &input_count), "vip_query_network(INPUT_COUNT)")
        _check(c.vip_query_network(self.handle, c.VIP_NETWORK_PROP_OUTPUT_COUNT,
                                    &output_count), "vip_query_network(OUTPUT_COUNT)")

        self.inputs = [_make_buffer_for_index(self.handle, i, True)
                        for i in range(input_count)]
        self.outputs = [_make_buffer_for_index(self.handle, i, False)
                         for i in range(output_count)]

        for i, buf in enumerate(self.inputs):
            _check(c.vip_set_input(self.handle, i, (<_Buffer> buf).handle),
                   "vip_set_input")
        for i, buf in enumerate(self.outputs):
            _check(c.vip_set_output(self.handle, i, (<_Buffer> buf).handle),
                   "vip_set_output")

    def input_info(self):
        """Lista de dicts com metadados de cada entrada (shape, dtype, nome, quantizacao)."""
        return [self._buffer_info(b) for b in self.inputs]

    def output_info(self):
        """Lista de dicts com metadados de cada saida (shape, dtype, nome, quantizacao)."""
        return [self._buffer_info(b) for b in self.outputs]

    def _buffer_info(self, _Buffer b):
        return {
            "name": b.name,
            "shape": b.shape,
            "dtype": np.dtype(b.dtype).name,
            "quant_format": b.quant_format,
            "tf_scale": b.tf_scale,
            "tf_zero_point": b.tf_zero_point,
        }

    def run(self, list input_arrays):
        """Copia `input_arrays` (list de np.ndarray) para os buffers de entrada,
        executa a rede de forma sincrona no VIP9000 e retorna os outputs como
        list de np.ndarray (na dtype nativa do tensor, sem desquantizar)."""
        if self.handle is NULL:
            raise VipError("rede ja foi fechada (close() chamado)")
        if len(input_arrays) != len(self.inputs):
            raise ValueError(
                f"esperado {len(self.inputs)} entrada(s), recebido {len(input_arrays)}")

        cdef _Buffer buf
        cdef void *dst
        cdef void *src
        cdef size_t addr
        cdef Py_ssize_t nbytes

        for i, a in enumerate(input_arrays):
            buf = self.inputs[i]
            arr = np.ascontiguousarray(a, dtype=buf.dtype)
            if tuple(arr.shape) != buf.shape:
                raise ValueError(
                    f"entrada {i}: shape {tuple(arr.shape)} != esperado {buf.shape}")
            dst = c.vip_map_buffer(buf.handle)
            if dst is NULL:
                raise VipError("vip_map_buffer retornou NULL para entrada")
            nbytes = arr.nbytes
            addr = arr.ctypes.data
            memcpy(dst, <void *> addr, nbytes)
            _check(c.vip_unmap_buffer(buf.handle), "vip_unmap_buffer(input)")
            _check(c.vip_flush_buffer(buf.handle, c.VIP_BUFFER_OPER_TYPE_FLUSH),
                   "vip_flush_buffer(input)")

        _check(c.vip_run_network(self.handle), "vip_run_network")

        results = []
        for buf in self.outputs:
            _check(c.vip_flush_buffer(buf.handle, c.VIP_BUFFER_OPER_TYPE_INVALIDATE),
                   "vip_flush_buffer(output)")
            src = c.vip_map_buffer(buf.handle)
            if src is NULL:
                raise VipError("vip_map_buffer retornou NULL para saida")
            size = c.vip_get_buffer_size(buf.handle)
            out = np.empty(buf.shape, dtype=buf.dtype)
            addr = out.ctypes.data
            memcpy(<void *> addr, src, min(size, out.nbytes))
            _check(c.vip_unmap_buffer(buf.handle), "vip_unmap_buffer(output)")
            results.append(out)

        return results

    def close(self):
        """Libera os buffers e destroi a rede no driver. Chamar antes de vip9000.destroy()."""
        for buf in self.inputs:
            (<_Buffer> buf)._destroy()
        for buf in self.outputs:
            (<_Buffer> buf)._destroy()
        self.inputs = []
        self.outputs = []
        self._nbg_data = None
        if self.handle is not NULL:
            if self._prepared:
                c.vip_finish_network(self.handle)
            c.vip_destroy_network(self.handle)
            self.handle = NULL

    def __dealloc__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
