import sys
import numpy as np
import vip9000


def main():
    nbg_path = sys.argv[1]
    vip9000.init()
    print("NPU_SW_VERSION (build):", vip9000.BUILT_NPU_SW_VERSION)
    print("VIPLite version:", vip9000.get_version())
    print("hardware:", vip9000.query_hardware())

    # Network aceita tanto um caminho (str/PathLike, via
    # VIP_CREATE_NETWORK_FROM_FILE) quanto o conteudo do .nb ja em
    # memoria (bytes/bytearray/memoryview/np.ndarray, via
    # VIP_CREATE_NETWORK_FROM_MEMORY). Aqui simulamos o caso de buffer
    # lendo o arquivo manualmente antes.
    if use_buffer:
        with open(nbg_path, "rb") as f:
            source = f.read()  # bytes -> carregado via buffer
        print(f"carregando NBG via buffer ({len(source)} bytes)")
    else:
        source = nbg_path      # str -> carregado via arquivo

    with vip9000.Network(source) as net:
        print("inputs :", net.input_info())
        print("outputs:", net.output_info())
        # gera entrada dummy com o shape/dtype esperado pela rede
        info = net.input_info()[0]
        inputs = [np.zeros(info["shape"], dtype=np.dtype(info["dtype"]))]
        outputs = net.run(inputs)
        for i, o in enumerate(outputs):
            print(f"output[{i}]: shape={o.shape} dtype={o.dtype} "
                    f"min={o.min()} max={o.max()}")
    vip9000.destroy()


if __name__ == "__main__":
    main()
