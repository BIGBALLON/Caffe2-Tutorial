
from torch.autograd import Variable
import torch.onnx
import torchvision

# Standard ImageNet input - 3 channels, 224x224,
# values don't matter as we care about network structure.
# But they can also be real inputs.
dummy_input = Variable(torch.randn(1, 3, 224, 224))
# Obtain your model, it can be also constructed in your script explicitly
model = torchvision.models.resnet50(pretrained=True)
model.eval()
# Invoke export
torch.onnx.export(model, dummy_input, "resnet50.onnx")
