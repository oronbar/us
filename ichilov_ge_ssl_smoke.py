"""Memory and gradient smoke test for the GE SSL partial MViT adaptation."""
import torch
from torch import nn

from dicom_prediction_encode import load_encoder
from ichilov_ge_ssl_train import encode, contrastive

torch.set_num_threads(4)
model = load_encoder("echoprime", __import__("pathlib").Path(r"D:\us\output\dicom_prediction"), "cuda")
model.eval().requires_grad_(False)
for module in (model.blocks[-1], model.norm, model.head):
    module.requires_grad_(True)
projector = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 128)).cuda()
order_head = nn.Linear(512, 2).cuda()
pixels = torch.randint(0, 256, (12, 3, 16, 224, 224), device="cuda", dtype=torch.uint8).float()
output = encode(model, pixels)
first, second, bad = output.chunk(3, dim=0)
loss, _ = contrastive(projector(first), projector(second), [str(i) for i in range(4)],
                      torch.empty((0, 128), device="cuda"), [])
logits = torch.cat((order_head(first), order_head(bad)))
targets = torch.cat((torch.zeros(4, dtype=torch.long, device="cuda"),
                     torch.ones(4, dtype=torch.long, device="cuda")))
loss = loss + .2 * nn.functional.cross_entropy(logits, targets)
loss.backward()
print({"output_shape": tuple(output.shape), "loss": float(loss.item()),
       "last_block_grad": float(next(model.blocks[-1].parameters()).grad.abs().mean()),
       "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
