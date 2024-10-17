import torch
import numpy as np
from torch import nn
from torch.autograd import Function

class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert n == m, "Point clouds must have the same number of points."
        assert xyz1.size(0) == xyz2.size(0), "Batch sizes must match."
        assert n % 1024 == 0, "Number of points must be a multiple of 1024."
        assert batchsize <= 512, "Batch size must be no greater than 512."

        # Ensure inputs are on the GPU and contiguous
        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()

        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.full((batchsize, n), -1, device='cuda', dtype=torch.int32)

        # Initializing cost and price matrices for auction algorithm
        price = torch.zeros(batchsize, n, device='cuda')
        cost = torch.zeros(batchsize, n, m, device='cuda')

        for _ in range(iters):
            for i in range(n):
                # Calculate pairwise distances
                point_diff = xyz1[:, i, :].unsqueeze(1) - xyz2
                cost[:, i, :] = torch.norm(point_diff, dim=2)

            # Adjust the cost by subtracting the current price
            cost_adjusted = cost - price.unsqueeze(2)

            # Find the minimum cost for each point in xyz1
            min_cost, min_idx = torch.min(cost_adjusted, dim=2)

            # Update assignments and prices based on the auction algorithm
            assignment[:, :] = min_idx
            price += eps * min_cost

        # Calculate the final distance
        ctx.save_for_backward(xyz1, xyz2, assignment)
        return torch.sqrt(min_cost), assignment

    @staticmethod
    def backward(ctx, graddist, _):
        xyz1, xyz2, assignment = ctx.saved_tensors
        gradxyz1 = torch.zeros_like(xyz1, device='cuda')
        gradxyz2 = torch.zeros_like(xyz2, device='cuda')

        for b in range(xyz1.size(0)):
            for i in range(xyz1.size(1)):
                j = assignment[b, i].item()
                if j >= 0:
                    gradxyz1[b, i, :] = 2 * (xyz1[b, i, :] - xyz2[b, j, :])

        return gradxyz1, None, None, None

class emdModule(nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, input1, input2, eps, iters):
        return emdFunction.apply(input1, input2, eps, iters)

def test_emd():
    x1 = torch.rand(20, 8192, 3).cuda()
    x2 = torch.rand(20, 8192, 3).cuda()
    emd = emdModule()
    start_time = time.perf_counter()
    dist, assignment = emd(x1, x2, 0.05, 3000)
    print("Input_size: ", x1.shape)
    print("Runtime: {:.6f}s".format(time.perf_counter() - start_time))
    print("EMD: {:.6f}".format(dist.mean().item()))
    print("|set(assignment)|: {}".format(torch.unique(assignment).numel()))
    assignment = assignment.cpu().numpy()
    assignment = np.expand_dims(assignment, -1)
    x2_aligned = np.take_along_axis(x2.cpu().numpy(), assignment, axis=1)
    d = (x1.cpu().numpy() - x2_aligned) ** 2
    verified_emd = np.sqrt(d.sum(-1)).mean()
    print("Verified EMD: {:.6f}".format(verified_emd))

# Uncomment to test the function
# test_emd()

        