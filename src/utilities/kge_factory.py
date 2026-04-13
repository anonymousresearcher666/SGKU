from typing import Dict

from src.model.kge_models.TransE import TransE
from src.model.kge_models.RotatE import RotatE
from src.model.kge_models.DistMult import DistMult
from src.model.kge_models.ComplexE import ComplexE


KGE_CLASS_MAP: Dict[str, type] = {
    "transe": TransE,
    "rotate": RotatE,
    "distmult": DistMult,
    "complexe": ComplexE,
}


def get_kge_model_class(name: str):
    key = (name or "transe").lower()
    if key not in KGE_CLASS_MAP:
        raise ValueError(f"Unsupported KGE model: {name}")
    return KGE_CLASS_MAP[key]
