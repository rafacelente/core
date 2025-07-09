import torch
import fr_norm


# torch fr norm
z = torch.randn(10, 10)
z.norm(p="fro")
print(z)

x = torch.randn(10, 10)
accum = torch.zeros(1)

fr_norm.fr_tk(x, accum)
print(accum)



