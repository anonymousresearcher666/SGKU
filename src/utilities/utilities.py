import datetime
import os
import random
from typing import Union

import numpy as np
import torch
import torch.backends.cudnn


def get_datetime():
    """Get current time."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def resolve_device(preferred: Union[str, torch.device, None] = None) -> torch.device:
    """
    Resolve the most appropriate torch.device given a preferred option.

    Args:
        preferred: Preferred device spec (e.g. "mps", "cuda", "cpu").

    Returns:
        torch.device: A device that is available on the current machine.
    """
    if isinstance(preferred, torch.device):
        target = preferred.type
    else:
        target = (preferred or "cpu").lower()

    if target.startswith("mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        # Fallback to CUDA if available, otherwise CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if target.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Default path covers "cpu" and unknown strings
    return torch.device("cpu")


def should_pin_memory(device: Union[str, torch.device, None]) -> bool:
    """
    Determine whether DataLoader pin_memory should be enabled for a given device.

    Args:
        device: Device specification.

    Returns:
        bool: True if using CUDA, False otherwise.
    """
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return device.lower().startswith("cuda")
    return False


# Utility function to calculate unlearning metrics
def calculate_unlearning_metrics(forget_processor, reserve_processor):
    """
    Calculate unlearning metrics aligned with the paper (MRR-based):

      - MRR_r: MRR on retain split (higher is better)
      - MRR_f: MRR on accumulated forget split (lower is better)
      - MRR_Avg = (MRR_r + (1 - MRR_f)) / 2
      - MRR_F1  = 2 * MRR_r * (1 - MRR_f) / (MRR_r + (1 - MRR_f))

    Args:
        forget_processor: Processor for the forgetting dataset
        reserve_processor: Processor for the reserved dataset

    Returns:
        dict: Dictionary with paper metrics and backward-compatible aliases.
    """
    # Get MRR values
    mrr_f = float(forget_processor.get_mrr_f())  # MRR_f: MRR on accumulated forget split
    mrr_r = float(reserve_processor.get_mrr_r())  # MRR_r: MRR on retain split

    # Calculate metrics
    one_minus_mrr_f = 1.0 - mrr_f

    # Paper: MRR_Avg
    mrr_avg = (mrr_r + one_minus_mrr_f) / 2.0

    # Paper: MRR_F1
    denominator = mrr_r + one_minus_mrr_f
    if denominator > 0:
        mrr_f1 = (2.0 * mrr_r * one_minus_mrr_f) / denominator
    else:
        mrr_f1 = 0.0

    # Print results
    print("\n=== Unlearning Performance Metrics (paper) ===")
    print(f"MRR_r: {mrr_r:.4f} (higher is better)")
    print(f"MRR_f: {mrr_f:.4f} (lower is better)")
    print(f"Forget Success (1-MRR_f): {one_minus_mrr_f:.4f}")
    print(f"MRR_Avg: {mrr_avg:.4f}")
    print(f"MRR_F1: {mrr_f1:.4f}")

    return {
        # Paper names (exact)
        "MRR_r": mrr_r,
        "MRR_f": mrr_f,
        "MRR_Avg": mrr_avg,
        "MRR_F1": mrr_f1,
        # Backward-compatible aliases used elsewhere in the repo
        "mrr_r": mrr_r,
        "mrr_f": mrr_f,
        "mrr_avg": mrr_avg,
        "mrr_f1": mrr_f1,
    }



def set_seeds(seed):
    """ Set  seeds """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    pass
