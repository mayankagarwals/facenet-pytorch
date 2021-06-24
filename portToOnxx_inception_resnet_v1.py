from models.inception_resnet_v1 import InceptionResnetV1
import torch 
import onnx
import onnxruntime
import numpy as np
import time


# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Input to the model
x = torch.randn(1,3,160,160, requires_grad = True)

start = time.time()

#for later verification
torch_out = resnet(x)

end = time.time()
print(f"Time taken by pytorch runtime : {end - start} s")


# Export the model
torch.onnx.export(resnet,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "inception_resnet_v1.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})



#validate graph  has the right schema 
onnx_model = onnx.load("inception_resnet_v1.onnx")
onnx.checker.check_model(onnx_model)

#validate output is same

ort_session = onnxruntime.InferenceSession("inception_resnet_v1.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

start = time.time()

ort_outs = ort_session.run(None, ort_inputs)

end = time.time()
print(f"Time taken by onnx runtime : {end - start} s")

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")


'''
Todo:
1. Time both versions
'''