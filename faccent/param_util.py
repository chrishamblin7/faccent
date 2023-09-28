import torch
import torch.nn.functional as F

def create_alpha_mask(shape, blur_radius):
    """
    Creates a mask with alpha blending effect.
    
    Args:
    - shape (tuple): Shape of the desired mask (c, h, w).
    - blur_radius (int): The radius to apply blur.
    
    Returns:
    - mask (torch.Tensor): The alpha mask tensor.
    """
    _, h, w = shape
    
    # Create a 1D tensor representing the distance of each row/column from the closest edge
    rows = torch.min(torch.arange(h).float(), (h - torch.arange(h).float()) - 1)
    cols = torch.min(torch.arange(w).float(), (w - torch.arange(w).float()) - 1)

    # Expand dims and combine to create a 2D distance map
    rows = rows[:, None]
    cols = cols[None, :]

    # Compute alpha values based on distance to edge and blur_radius
    distance_map = torch.min(rows, cols)
    alpha_map = torch.clamp(distance_map / float(blur_radius), 0, 1)
    
    # Expand to match channel dimension
    mask = alpha_map.expand(shape[0], -1, -1)
    
    return mask



def center_paste(x, y, blur_radius=0):
    """
    Pastes tensor x onto the center of tensor y with an optional blur on the boundary.
    
    Args:
    - x (torch.Tensor): The tensor to be pasted. Shape (c, h, w).
    - y (torch.Tensor): The larger tensor where x will be pasted. Shape (c, h2, w2).
    - blur_radius (int): The radius to apply blur. A value of 0 means sharp boundary.
    
    Returns:
    - z (torch.Tensor): Tensor y with x pasted in the center. Shape (c, h2, w2).
    """
    
    # Ensure input tensors have the correct shapes
    if x.dim() == 4: x = x.squeeze()
    if y.dim() == 4: y = y.squeeze() 
    assert len(x.shape) == 3, "x should have shape (c, h, w)"
    assert len(y.shape) == 3, "y should have shape (c, h2, w2)"
    assert x.shape[0] == y.shape[0], "Both tensors should have the same number of channels"
    
    y = y.to(x.device)
    
    # Calculate start and end positions for height and width
    start_h = (y.shape[1] - x.shape[1]) // 2
    end_h = start_h + x.shape[1]
    
    start_w = (y.shape[2] - x.shape[2]) // 2
    end_w = start_w + x.shape[2]

    # Create the resulting tensor z
    z = y.clone()

    if blur_radius <= 0:
        z[:, start_h:end_h, start_w:end_w] = x
    else:
        mask = create_alpha_mask(x.shape, blur_radius)
        mask = mask.to(x.device)
        # Create a linear mask for alpha blending
        #mask = torch.ones_like(x).to(x.device)
        #for i in range(blur_radius):
        #   mask = create_alpha_mask(x.shape, blur_radius)
            #alpha = i / float(blur_radius)
            #mask[:, i, :] = mask[:, x.shape[1] - i - 1, :] = alpha
            #mask[:, :, i] = mask[:, :, x.shape[2] - i - 1] = alpha
        
        # Apply the mask using linear blending
        z[:, start_h:end_h, start_w:end_w] = mask * x + (1 - mask) * z[:, start_h:end_h, start_w:end_w]

    return z.unsqueeze(dim=0)