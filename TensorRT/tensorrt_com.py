import onnx
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import time
import torchvision
import numpy as np
import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
engine = None


def Init_TensorRT(trt_path):
    global engine
    engine = load_engine(trt_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(
        engine
    )  # input, output: host # bindings
    return [context, inputs, outputs, bindings, stream]


def load_engine(trt_path):

    TRT_LOGGER = trt.Logger()
    if os.path.exists(trt_path):
        print("Reading engine from file: {}".format(trt_path))
        with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("No Found:" + trt_path)
        raise FileNotFoundError


def allocate_buffers(engine):
    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            """
            host_mem: cpu memory
            device_mem: gpu memory
            """
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        # append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def Do_Inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # gpu to cpu
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def Torch_to_ONNX(net, input_size, onnx_model_path, device):
    net.to(device)
    net.eval()
    torch.onnx.export(
        net,
        torch.randn(tuple(input_size), device=device),
        onnx_model_path,
        verbose=False,
        input_names=["input"] + ["params_%d" % i for i in range(120)],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
    )

    # import onnx
    net = onnx.load(onnx_model_path)
    onnx.checker.check_model(net)
    onnx.helper.printable_graph(net.graph)

    # ONNX
    import onnxruntime

    session = onnxruntime.InferenceSession(onnx_model_path)
    out_r = session.run(
        None,
        {
            "input": np.random.rand(
                input_size[0], input_size[1], input_size[2], input_size[3]
            ).astype("float32")
        },
    )
    print("ONNX file in " + onnx_model_path)
    print("============Pytorch->ONNX SUCCESS============")


def ONNX_to_TensorRT(
    max_batch_size=1, fp16_mode=False, onnx_model_path=None, trt_engine_path=None
):
    TRT_LOGGER = trt.Logger()

    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # In TensorRT 7.0, the ONNX parser only supports full-dimensions mode, meaning that your network definition must be created with the explicitBatch flag set. For more information, see Working With Dynamic Shapes.

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        explicit_batch
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = max_batch_size
        builder.fp16_mode = fp16_mode

        if not os.path.exists(onnx_model_path):
            quit("ONNX file {} not found!".format(onnx_model_path))
        print("loading onnx file from path {} ...".format(onnx_model_path))
        with open(onnx_model_path, "rb") as model:
            print("Begining onnx file parsing")
            parser.parse(model.read())
        # parser.parse_from_file(onnx_model_path)

        print("Completed parsing of onnx file")
        # CudaEngine
        print(
            "Building an engine from file{}' this may take a while...".format(
                onnx_model_path
            )
        )

        #################
        output_shape = network.get_layer(network.num_layers - 1).get_output(0).shape
        # network.mark_output(network.get_layer(network.num_layers -1).get_output(0))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")

        #
        with open(trt_engine_path, "wb") as f:
            f.write(engine.serialize())

        print("TensorRT file in " + trt_engine_path)
        print("============ONNX->TensorRT SUCCESS============")
