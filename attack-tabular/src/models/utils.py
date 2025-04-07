import logging
import torch

from src.models.mlp import LitMLP

logger = logging.getLogger(__name__)


def load_trained_model(saved_model_path, model_type='mlp',
                       dataset=None, eval_mode=True):
    """
    Loads the required model from the saved models directory
    """
    logger.info(f"Loading model from {saved_model_path}")

    if model_type == 'mlp':
        map_location = None if torch.cuda.is_available() else 'cpu'
        model = LitMLP.load_from_checkpoint(saved_model_path, map_location=map_location)
    # elif model_type == 'tabnet':  # TODO [ADD-FEATURE] support tabnet
    #     assert dataset is not None, "dataset properties must be provided for TabNet model"
    #     model_wrapper = TabNetModelWrapper.load_model(model_ref_zip=saved_model_path,
    #                                                   cat_idxs=dataset.unordered_indices,
    #                                                   cat_dims=dataset.unordered_dims)
    #     model = model_wrapper.get_model_object()
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    if eval_mode:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model

