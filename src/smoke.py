import torch
import blocksparse_cuda

grad = torch.randn(10, dtype=torch.float16, device='cuda')
loss = torch.randn(10, dtype=torch.float32, device='cuda')
logits = torch.randn(10, dtype=torch.float16, device='cuda')
labels = torch.randint(0, 5, (10, ), dtype=torch.int32, device='cuda')

blocksparse_cuda.forward(grad, loss, logits, labels, 1, 1)

dlogits = torch.zeros(10, dtype=torch.float32, device='cuda')
dx = torch.zeros(10, dtype=torch.float16, device='cuda')
blocksparse_cuda.backward(dx, dlogits, logits, 1, 1)
