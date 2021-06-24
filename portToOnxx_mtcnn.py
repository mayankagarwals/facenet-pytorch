from models.mtcnn import MTCNN
import torch 
import onnx
import onnxruntime
import numpy as np
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Create an mtcnn model (in eval mode):
mtcnn = MTCNN(device=device)

# Input to the model
x = torch.randn(480, 640, 3, requires_grad = True)

start = time.time()

#for timing difference
torch_out = mtcnn.detect(x ,landmarks = True)

end = time.time()
print(f"Time taken by pytorch runtime : {end - start} s")


# Export the model
torch.onnx.export(mtcnn,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "mtcnn.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'height',
                                          1: 'width'}}
                                          )    # variable length axes})



# #validate graph  has the right schema 
# onnx_model = onnx.load("inception_resnet_v1.onnx")
# onnx.checker.check_model(onnx_model)

# #time output 

# ort_session = onnxruntime.InferenceSession("mtcnn.onnx")

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

# start = time.time()

# ort_outs = ort_session.run(None, ort_inputs)

# end = time.time()
# print(f"Time taken by onnx runtime : {end - start} s")

