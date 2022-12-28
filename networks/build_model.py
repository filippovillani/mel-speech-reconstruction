import torch

from networks.PInvConv.models import PInv, PInvConv
from networks.UNet.models import UNet
from networks.DeGLI.layers import DeGLIBlock

def build_model(hparams,
                model_name,
                weights_dir = None,
                best_weights: bool = True):
    
    if model_name.lower() == "unet":
        model = UNet(hparams).float().to(hparams.device)
    elif model_name.lower() == "convpinv":
        model = PInvConv(hparams).float().to(hparams.device)
    elif model_name.lower() == "pinv":
        model = PInv(hparams).float().to(hparams.device)
    else:
        raise ValueError(f"model_name must be one of [unet, convpinv, pinv], received: {str(model_name)}")
        
    if weights_dir is not None and model_name != "pinv":
        weights_path = (weights_dir / 'best_weights') if best_weights else (weights_dir / 'ckpt_weights')
        model.load_state_dict(torch.load(weights_path))
    
    return model 