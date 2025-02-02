import torch
import time
device = "cuda" if torch.cuda.is_available() else "cpu"

pose_model = torch.jit.load("saved_models/pose.pt")
pose_model.to(device)

for i in range(20):
    test_input = torch.rand((1, 7, 256, 256)).to(device)
    start_body = time.time()
    body_output = pose_model.sample_body(test_input)
    convert_to_cpu = body_output[0][0].item()
    end_body = time.time()

    start_head = time.time()
    head_output = pose_model.sample_head(body_output)
    convert_to_cpu = head_output[0].to("cpu")
    end_head = time.time()

    print(f"Body Time: {end_body-start_body}")
    print(f"Head Time: {end_head-start_head}")
