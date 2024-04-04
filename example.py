import torch
from fused_qat_linear import QLinear

# Define the dimensions of the input and output layers
clip_val = 4.0
clip_valn  = 3.0
num_bits = 4

batch_size=1
seq_len=2048
input_dim = 1024
output_dim = 2048

# Instantiate the QLinear layer
linear_layer = QLinear(clip_val, clip_valn, num_bits, input_dim, output_dim).cuda().half()

# Create some random input data
input_data = torch.randn(batch_size, seq_len, input_dim).cuda().half()

# Pass the input data through the linear layer
output_data = linear_layer(input_data)

# Print the output
print(output_data)