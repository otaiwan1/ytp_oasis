import torch
import torch.nn as nn
import torch.nn.functional as F

def get_knn(x, k):
    """
    Calculates the k-nearest neighbors for a point cloud.
    Args:
        x: (batch_size, num_dims, num_points) - Input features/coords
            batch: number of objects; dimension (channels): 3 -> 64 -> 1024...; points: number of points in the cloud
        k: int - Number of neighbors
    Returns:
        idx: (batch_size, num_points, k) - Indices of the nearest neighbors
    """
    batch_size = x.size(0)
    num_points = x.size(2)

    # 1. Calculate Pairwise Distance
    # Formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    x = x.transpose(2, 1) # (B, C, N) -> (B, N, C)
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    # 2. Find Top-k (closest)
    # We use topk on the negative distance to find the smallest distances
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    
    return idx

def get_graph_feature(x, k, idx=None):
    """
    Constructs the local graph features for EdgeConv.
    Args:
        x: (batch_size, num_dims, num_points)
        k: int
        idx: neighbors indices (optional, can be precomputed)
    Returns:
        feature: (batch_size, 2*num_dims, num_points, k)
    """
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    # If indices not provided, compute k-NN
    if idx is None:
        idx = get_knn(x, k) # (batch_size, num_points, k)

    # Prepare indices for gathering
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    # Gather neighbor features
    x_flat = x.transpose(2, 1).contiguous().view(batch_size * num_points, num_dims)
    feature = x_flat[idx, :] 
    feature = feature.view(batch_size, num_points, k, num_dims)
    
    # Permute to (Batch, Dims, Points, k)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    # Result shape: (Batch, 2*Channels, Num_Points, k)
    return feature

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        
        # The "MLP" is implemented as a Conv2d because our data is 
        # structured as (Batch, Channels, Points, Neighbors)
        # Input channels are doubled because we concat (x_j - x_i) and x_i
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        # x shape: (batch_size, in_channels, num_points)
        
        # 1. Construct the graph feature
        # Shape: (batch_size, 2*in_channels, num_points, k)
        x_graph = get_graph_feature(x, k=self.k)
        
        # 2. Apply the MLP (h_theta)
        # Shape: (batch_size, out_channels, num_points, k)
        x_graph = self.conv(x_graph)
        
        # 3. Aggregate (Max Pooling) over the neighbor dimension (dim 3)
        # Shape: (batch_size, out_channels, num_points)
        x_out = x_graph.max(dim=-1, keepdim=False)[0]
        
        return x_out



# Example Usage
# Batch size 2, 3 coordinates (x,y,z), 1024 points
input_points = torch.rand(2, 3, 1024) 

# Initialize EdgeConv
# Input dim 3 -> Output dim 64, using 20 nearest neighbors
edge_conv_layer = EdgeConv(in_channels=3, out_channels=64, k=20)

# Forward pass
high_dim_vectors = edge_conv_layer(input_points)

print(f"Input shape: {input_points.shape}")   # torch.Size([2, 3, 1024])
print(f"Output shape: {high_dim_vectors.shape}") # torch.Size([2, 64, 1024])