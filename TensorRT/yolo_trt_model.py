import os
import sys
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
from tensorrt_com import (
    Do_Inference,
    Init_TensorRT,
    ONNX_to_TensorRT,
)


class YoloTrtModel:
    def __init__(self, device_id="cuda:0", trt_engine_path=None, fp16_mode=False):

        self.cfx = cuda.Device(0).make_context()
        self.model_params = Init_TensorRT(trt_engine_path)

        self.stride8_shape = (1, 3, 40, 40, 16)
        self.stride16_shape = (1, 3, 20, 20, 16)
        self.stride32_shape = (1, 3, 10, 10, 16)

    def __call__(self, img_np_nchw):
        context, inputs, outputs, bindings, stream = self.model_params
        self.cfx.push()

        inputs[0].host = img_np_nchw.reshape(-1)
        trt_outputs = Do_Inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )  # numpy data
        stride_8 = trt_outputs[0].reshape(*self.stride8_shape)
        stride_16 = trt_outputs[1].reshape(*self.stride16_shape)
        stride_32 = trt_outputs[2].reshape(*self.stride32_shape)
        self.cfx.pop()
        return [stride_8, stride_16, stride_32]

    def after_process(self, pred, device):
        stride = torch.tensor([8.0, 16.0, 32.0]).to(device)

        x = [
            torch.from_numpy(pred[0]).to(device),
            torch.from_numpy(pred[1]).to(device),
            torch.from_numpy(pred[2]).to(device),
        ]

        no = 16
        nl = 3

        grid = [torch.zeros(1).to(device)] * nl

        anchor_grid = torch.tensor(
            [
                [[[[[4.0, 5.0]]], [[[8.0, 10.0]]], [[[13.0, 16.0]]]]],
                [[[[[23.0, 29.0]]], [[[43.0, 55.0]]], [[[73.0, 105.0]]]]],
                [[[[[146.0, 217.0]]], [[[231.0, 300.0]]], [[[335.0, 433.0]]]]],
            ]
        ).to(device)

        z = []
        for i in range(len(x)):
            bs, ny, nx = x[i].shape[0], x[i].shape[2], x[i].shape[3]
            if grid[i].shape[2:4] != x[i].shape[2:4]:
                grid[i] = self._make_grid(nx, ny).to(device)
            y = torch.full_like(x[i], 0)
            y[..., [0, 1, 2, 3, 4, 15]] = x[i][..., [0, 1, 2, 3, 4, 15]].sigmoid()
            y[..., 5:15] = x[i][..., 5:15]

            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid[i].to(device)) * stride[
                i
            ]  # xy

            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh

            y[..., 5:7] = (
                y[..., 5:7] * anchor_grid[i] + grid[i].to(device) * stride[i]
            )  # landmark x1 y1

            y[..., 7:9] = (
                y[..., 7:9] * anchor_grid[i] + grid[i].to(device) * stride[i]
            )  # landmark x2 y2

            y[..., 9:11] = (
                y[..., 9:11] * anchor_grid[i] + grid[i].to(device) * stride[i]
            )  # landmark x3 y3

            y[..., 11:13] = (
                y[..., 11:13] * anchor_grid[i] + grid[i].to(device) * stride[i]
            )  # landmark x4 y4

            y[..., 13:15] = (
                y[..., 13:15] * anchor_grid[i] + grid[i].to(device) * stride[i]
            )  # landmark x5 y5

            z.append(y.view(bs, -1, no))

        return torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
