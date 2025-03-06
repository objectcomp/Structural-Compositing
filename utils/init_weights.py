import sys
if './' not in sys.path:
    sys.path.append('./')

import torch

from utils.share import *
from models.util import create_model

def init_comp(sd_weights_path, config_path, output_path):
    """
    Example: loads a base SD checkpoint and maps certain layers to your 'comp' model.
    If you have different logic in mind, adjust accordingly.
    """
    pretrained_weights = torch.load(sd_weights_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    # Create your model from the config
    model = create_model(config_path=config_path)

    # Initialize a dictionary for final parameters
    scratch_dict = model.state_dict()
    target_dict = {}

    # For each parameter name in your new model,
    # see if there's a matching key in the SD checkpoint (after rewriting).
    for sk in scratch_dict.keys():
        # Example: if your model had "local_adapter." we rename to "model.diffusion_model."
        # Adjust this rename logic if your architecture differs.
        old_k = sk.replace('local_adapter.', 'model.diffusion_model.')

        if old_k in pretrained_weights:
            target_dict[sk] = pretrained_weights[old_k].clone()
        else:
            # If no match, just keep your newly inited parameter
            target_dict[sk] = scratch_dict[sk].clone()
            print(f'new params (no match in SD weights): {sk}')

    # Load the new state dict
    model.load_state_dict(target_dict, strict=True)
    # Save to output
    torch.save(model.state_dict(), output_path)
    print('Done init_comp ->', output_path)


if __name__ == '__main__':
    # We want exactly 5 argv items: 
    #   sys.argv[0] = "prepare_weights.py"
    #   sys.argv[1] = "init_comp"
    #   sys.argv[2] = <sd_weights_path>
    #   sys.argv[3] = <config_path>
    #   sys.argv[4] = <output_path>
    if len(sys.argv) != 5:
        print("Usage: python utils/prepare_weights.py init_comp <sd_ckpt> <config> <out_ckpt>")
        sys.exit(1)

    mode = sys.argv[1]
    sd_ckpt = sys.argv[2]
    config_path = sys.argv[3]
    out_ckpt = sys.argv[4]

    if mode != 'init_comp':
        raise ValueError(f"Mode must be 'init_comp', got {mode}")

    init_comp(sd_ckpt, config_path, out_ckpt)
