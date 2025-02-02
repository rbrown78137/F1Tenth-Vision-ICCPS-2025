import torch
from YOLOV3Tiny import YOLOV3Tiny

model = YOLOV3Tiny()
model.load_state_dict(torch.load("saved_models/train_network.pth"))
model.eval()
script_model = torch.jit.script(model)
script_model.save("saved_models/yolo.pt")
print("Done")
