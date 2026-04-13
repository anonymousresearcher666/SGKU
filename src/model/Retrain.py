from torch import nn

from src.utilities.kge_factory import get_kge_model_class


class Retrain(nn.Module):
    """
    Generic retraining wrapper that instantiates the requested KGE model
    and simply delegates loss/predict calls.
    """

    def __init__(self, args, kg) -> None:
        super().__init__()
        self.args = args
        self.kg = kg
        kge_class = get_kge_model_class(getattr(args, "kge", "transe"))
        self.kge_model = kge_class(args, kg)

    def loss(self, head, relation, tail=None, label=None):
        return self.kge_model.loss(head, relation, tail, label)

    def predict(self, head, relation):
        return self.kge_model.predict(head, relation)

    def forward(self, *inputs, **kwargs):
        if hasattr(self.kge_model, "forward"):
            return self.kge_model.forward(*inputs, **kwargs)
        raise NotImplementedError("Underlying KGE model does not implement forward")
