import torch

def compute_loss(model: torch.nn.Module, model_output: torch.tensor, true_pose: torch.tensor):
    """
    Computes the loss between the model output and the true pose values

    Args:
    -   model: model (which inherits from nn.Module), and contains loss_criterion
    -   model_output: the predicted tool pose based on the model [Dim: (N,6)]
    -   true_pose: the ground truth pose [Dim: (N,6)]
    -   is_normalize: bool flag indicating that loss should be divided by the
                      batch size
    Returns:
    -   the loss value
    """

    loss = model.loss_criterion(model_output,true_pose)

    return loss