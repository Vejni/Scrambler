"""PyTorch port of Scrambler networks (Linder et al., Nat. Mach. Intell. 2022)."""
from .losses import get_entropy_loss, get_nll_loss
from .network import ScramblerNet, scale_logits
from .sampler import PSSMSampler, sample_pssm
from .scrambler import DEFAULT_NETWORK_CONFIG, Objective, Scrambler, ScramblerOutput
from .utils import background_from_data, one_hot_encode, sequence_mask, sequence_template

__all__ = ["Scrambler", "ScramblerNet", "ScramblerOutput", "Objective", "PSSMSampler", "sample_pssm", "scale_logits",
           "get_nll_loss", "get_entropy_loss", "one_hot_encode", "sequence_template", "sequence_mask",
           "background_from_data", "DEFAULT_NETWORK_CONFIG"]
