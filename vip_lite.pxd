cdef extern from "vip_lite.h":
    ctypedef unsigned int vip_uint32_t
    ctypedef unsigned char vip_uint8_t
    ctypedef int vip_int32_t
    ctypedef int vip_status_e
    ctypedef int vip_enum
    ctypedef char vip_char_t
    ctypedef float vip_float_t
    ctypedef void* vip_ptr

    ctypedef struct _vip_network:
        pass
    ctypedef _vip_network* vip_network

    ctypedef struct _vip_buffer:
        pass
    ctypedef _vip_buffer* vip_buffer

    ctypedef struct _vip_group:
        pass
    ctypedef _vip_group* vip_group

    cdef enum:
        VIP_SUCCESS
        VIP_ERROR_FAILURE
        VIP_ERROR_TIMEOUT
        VIP_ERROR_IO
        VIP_ERROR_INVALID_ARGUMENTS
        VIP_ERROR_NOT_SUPPORTED
        VIP_ERROR_OUT_OF_RESOURCE
        VIP_ERROR_OUT_OF_MEMORY
        VIP_ERROR_INVALID_NETWORK
        VIP_ERROR_MISSING_INPUT_OUTPUT
        VIP_ERROR_NETWORK_NOT_PREPARED
        VIP_ERROR_NETWORK_INCOMPATIBLE

    ctypedef enum vip_buffer_format_e:
        VIP_BUFFER_FORMAT_FP32
        VIP_BUFFER_FORMAT_FP16
        VIP_BUFFER_FORMAT_UINT8
        VIP_BUFFER_FORMAT_INT8
        VIP_BUFFER_FORMAT_UINT16
        VIP_BUFFER_FORMAT_INT16
        VIP_BUFFER_FORMAT_CHAR
        VIP_BUFFER_FORMAT_BFP16
        VIP_BUFFER_FORMAT_INT32
        VIP_BUFFER_FORMAT_UINT32
        VIP_BUFFER_FORMAT_INT64
        VIP_BUFFER_FORMAT_UINT64
        VIP_BUFFER_FORMAT_FP64

    ctypedef enum vip_buffer_quantize_format_e:
        VIP_BUFFER_QUANTIZE_NONE
        VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT
        VIP_BUFFER_QUANTIZE_TF_ASYMM

    ctypedef enum vip_create_network_type_e:
        VIP_CREATE_NETWORK_FROM_FILE
        VIP_CREATE_NETWORK_FROM_MEMORY
        VIP_CREATE_NETWORK_FROM_FLASH

    ctypedef enum vip_buffer_operation_type_e:
        VIP_BUFFER_OPER_TYPE_FLUSH
        VIP_BUFFER_OPER_TYPE_INVALIDATE

    ctypedef enum vip_network_property_e:
        VIP_NETWORK_PROP_LAYER_COUNT
        VIP_NETWORK_PROP_INPUT_COUNT
        VIP_NETWORK_PROP_OUTPUT_COUNT
        VIP_NETWORK_PROP_NETWORK_NAME

    ctypedef enum vip_buffer_property_e:
        VIP_BUFFER_PROP_QUANT_FORMAT
        VIP_BUFFER_PROP_NUM_OF_DIMENSION
        VIP_BUFFER_PROP_SIZES_OF_DIMENSION
        VIP_BUFFER_PROP_DATA_FORMAT
        VIP_BUFFER_PROP_FIXED_POINT_POS
        VIP_BUFFER_PROP_TF_SCALE
        VIP_BUFFER_PROP_TF_ZERO_POINT
        VIP_BUFFER_PROP_NAME

    ctypedef enum vip_query_hardware_property_e:
        VIP_QUERY_HW_PROP_CID
        VIP_QUERY_HW_PROP_DEVICE_COUNT
        VIP_QUERY_HW_PROP_CORE_COUNT_EACH_DEVICE

    ctypedef struct vip_buffer_create_params_t:
        vip_uint32_t num_of_dims
        vip_uint32_t sizes[6]
        vip_enum data_format
        vip_enum quant_format
        vip_uint32_t memory_type

    vip_uint32_t vip_get_version()
    vip_status_e vip_init()
    vip_status_e vip_destroy()
    vip_status_e vip_query_hardware(vip_query_hardware_property_e property,
                                     vip_uint32_t size, void *value)

    vip_status_e vip_create_network(const void *data, vip_uint32_t size_of_data,
                                     vip_create_network_type_e type,
                                     vip_network *network)
    vip_status_e vip_destroy_network(vip_network network)
    vip_status_e vip_set_network(vip_network network, vip_enum property, void *value)
    vip_status_e vip_query_network(vip_network network, vip_enum property, void *value)
    vip_status_e vip_prepare_network(vip_network network)
    vip_status_e vip_finish_network(vip_network network)

    vip_status_e vip_query_input(vip_network network, vip_uint32_t index,
                                  vip_enum property, void *value)
    vip_status_e vip_query_output(vip_network network, vip_uint32_t index,
                                   vip_enum property, void *value)

    vip_status_e vip_set_input(vip_network network, vip_uint32_t index, vip_buffer input)
    vip_status_e vip_set_output(vip_network network, vip_uint32_t index, vip_buffer output)

    vip_status_e vip_run_network(vip_network network)
    vip_status_e vip_trigger_network(vip_network network)
    vip_status_e vip_wait_network(vip_network network)
    vip_status_e vip_cancel_network(vip_network network)

    vip_status_e vip_create_buffer(vip_buffer_create_params_t *create_param,
                                    vip_uint32_t size_of_param, vip_buffer *buffer)
    vip_status_e vip_destroy_buffer(vip_buffer buffer)
    void *vip_map_buffer(vip_buffer buffer)
    vip_status_e vip_unmap_buffer(vip_buffer buffer)
    vip_uint32_t vip_get_buffer_size(vip_buffer buffer)
    vip_status_e vip_flush_buffer(vip_buffer buffer, vip_buffer_operation_type_e type)
