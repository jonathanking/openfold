from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class EarlyStoppingVerbose(EarlyStopping):
    """
        The default EarlyStopping callback's verbose mode is too verbose.
        This class outputs a message only when it's getting ready to stop. 
    """
    def __init__(self, min_steps=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_steps = min_steps

    def _evalute_stopping_criteria(self, *args, **kwargs):
        # Check if global step is less than min_steps
        if self.trainer.global_step < self.min_steps:
            return False, None

        should_stop, reason = super()._evalute_stopping_criteria(*args, **kwargs)
        if should_stop:
            rank_zero_info(f"{reason}\n")

        return should_stop, reason