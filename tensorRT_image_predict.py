import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# Load the TensorRT engine
engine_path = '/home/dev/Documents/Silent-Face-Anti-Spoofing/onnx/dynamic_trt/2.7_80x80_MiniFASNetV2.trt'
with open(engine_path, 'rb') as f:
    engine_data = f.read()
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)

# Get the input and output binding indices
input_index = engine.get_binding_index("images")  # Replace with your input name
output_index = engine.get_binding_index("output0")  # Replace with your output name

# Determine input and output shapes
input_shape = (1, 3, 80, 80)  # Replace with your input shape
output_shape = (1, 3)    # Replace with your output shape

# Create CUDA stream
stream = cuda.Stream()

while True:
    input_data = np.random.rand(*input_shape).astype(np.float32)

    # Allocate device memory for input
    d_input = cuda.mem_alloc(input_data.nbytes)

    # Calculate output size in bytes
    output_size = np.prod(output_shape) * 4  # 4 bytes per float32

    # Convert output_size to Python int and allocate device memory for output
    d_output = cuda.mem_alloc(int(output_size))


    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, input_data, stream)

    # Execute inference
    context = engine.create_execution_context()
    context.execute_v2(bindings=[int(d_input), int(d_output)])

    # Transfer output data to host
    output_data = np.empty(output_shape, dtype=np.float32)
    print(output_data.shape)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()

    # Process output data here (if necessary)

    # Release CUDA memory
    d_input.free()
    d_output.free()

    # Break condition or some kind of control to exit the loop
    # (not implemented here, but you should add it as needed)

# Release resources (not reached in this example due to the infinite loop)
context.destroy()
engine.destroy()
