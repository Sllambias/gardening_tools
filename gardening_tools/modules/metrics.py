from torch import Tensor, argmax
from torchmetrics.segmentation import GeneralizedDiceScore as TorchGeneralizedDiceScore


class GeneralizedDiceScore(TorchGeneralizedDiceScore):
    """
    Generalized Dice Score metric adapted to accept logits as input (preds) and target of any dtype.
    """

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = argmax(preds, dim=1)
        return super().update(preds, target.long())
