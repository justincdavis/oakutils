import torch


def depth_to_3d(depth: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    # depth should come in Bx1xHxW
    points_depth: torch.Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1
    points_3d: torch.Tensor = xyz * points_depth
    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW

class PointCloud(torch.nn.Module):
    def forward(self, xyz, depth):
        # TODO: once U16 -> FP16 is supported, use that.
        depthFP16 = 256.0 * depth[:,:,:,1::2] + depth[:,:,:,::2]
        return depth_to_3d(depthFP16, xyz)
