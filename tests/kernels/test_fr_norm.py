import torch

def test_fr_norm():
    torch.manual_seed(0)
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    z = x + y
    z.norm(p="fro")

