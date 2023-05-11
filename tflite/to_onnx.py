#!/usr/bin/python3

import torch

# Load the ESRGAN model
model = torch.load('model.pth')

# Set the input shape for the model
batch_size = 1
input_shape = (3, 400, 400)

# Create a dummy input tensor
dummy_input = torch.randn(batch_size, *input_shape)

# Export the PyTorch model to ONNX format
onnx_model_path = "model.onnx"
torch.onnx.export(model, dummy_input, onnx_model_path)
