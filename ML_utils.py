import torch

def compute_loss(model: torch.nn.Module, model_output: torch.tensor, true_pose: torch.tensor):
    """
    Computes the loss between the model output and the true pose values

    Args:
    -   model: model (which inherits from nn.Module), and contains loss_criterion
    -   model_output: the predicted tool pose based on the model [Dim: (N,6)]
    -   true_pose: the ground truth pose [Dim: (N,6)]
    Returns:
    -   the loss value
    """
    tmp_output = model_output
    loss = model.loss_criterion(model_output,true_pose)

    return loss

def compute_mean_and_std(dir_name: str):
    print("ML_tracking_model compute_mean_and_std")
    scaler = StandardScaler()
    for frame, _, _ in yield_segmentation_data(dir_name):
    #paths = glob.glob(dir_name+"/*/*/Video.avi")
    #for path in paths:
        pixels = np.reshape(np.array(frame),(-1,1))#list(Image.open(path).convert(mode="L").getdata())),(-1,1))
        normalized_pixels = np.divide(pixels,255.0)
        scaler.partial_fit(normalized_pixels)
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)
    return mean, std