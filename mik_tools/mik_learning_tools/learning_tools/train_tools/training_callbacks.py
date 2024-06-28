from pytorch_lightning import Callback
from tqdm import tqdm


class NormalizationQuickstart(Callback):
    """
    Start with a good estimate of the normalization constants.
    """
    def __init__(self, num_forward_passes=50):
        super().__init__()
        self.num_forward_passes = num_forward_passes

    def on_train_start(self, trainer, pl_module):
        train_loader = trainer.train_dataloader
        pbar = tqdm(range(self.num_forward_passes), desc='Normalization Quickstart')
        for i in pbar:
            for b_i in train_loader:
                # states_i = b_i['states'].to(pl_module.device)
                # _ = pl_module.update_normalization(states_i)
                _ = pl_module.update_normalization(b_i)