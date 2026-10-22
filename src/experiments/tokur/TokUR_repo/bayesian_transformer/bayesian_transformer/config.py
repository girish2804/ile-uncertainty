from transformers import PretrainedConfig, AutoConfig
from typing import List

class BayesianLMConfig(PretrainedConfig):
    model_type = "bayesian_lm"

    def __init__(
        self,
        base_lm_path: str = None,
        target_modules: List[str] = None,
        bayes_sigma: float = 1e-1,
        bayes_noise: str = 'right',
        basis_idx: tuple = (0,),
        sample: bool = True,
        num_samples: int = 10,
        lowrank: bool = True,
        **kwargs,
    ):
        if bayes_noise not in ["right", "left", "std", "self-scaling"]:
            raise ValueError(f"`bayes_noise` must be either 'right' or 'left', got {bayes_noise}.")
                
        # now the Bayesian Config only stores the path of the base_lm_config.
        self.base_lm_path = base_lm_path
        self.target_modules = target_modules
        self.bayes_sigma = bayes_sigma
        self.bayes_noise = bayes_noise
        self.basis_idx = basis_idx
        self.sample = sample
        self.num_samples = num_samples
        self.lowrank = lowrank

        super().__init__(**kwargs)