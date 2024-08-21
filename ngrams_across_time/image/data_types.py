from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset

from concept_erasure import QuadraticEditor, QuadraticFitter
from concept_erasure.quantile import QuantileNormalizer
from concept_erasure.utils import assert_type


@dataclass
class ConceptEditedDataset:
    class_probs: Tensor
    editor: QuadraticEditor
    X: Tensor
    Y: Tensor

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x, y = self.X[idx], int(self.Y[idx])

        # Make sure we don't sample the correct class
        loo_probs = self.class_probs.clone()
        loo_probs[y] = 0
        target_y = torch.multinomial(loo_probs, 1).squeeze()

        x = self.editor.transport(x[None], y, int(target_y)).squeeze(0)
        return {
            "pixel_values": x,
            "label": target_y,
        }

    def __len__(self) -> int:
        return len(self.Y)


@dataclass
class QuantileNormalizedDataset:
    class_probs: Tensor
    editor: QuantileNormalizer
    X: Tensor
    Y: Tensor

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x, y = self.X[idx], self.Y[idx]

        # Make sure we don't sample the correct class
        loo_probs = self.class_probs.clone()
        loo_probs[y] = 0
        target_y = torch.multinomial(loo_probs, 1).squeeze()

        lut1 = self.editor.lut[y]
        lut2 = self.editor.lut[target_y]

        indices = torch.searchsorted(lut1, x[..., None]).clamp(0, lut1.shape[-1] - 1)
        x = lut2.gather(-1, indices).squeeze(-1)

        return {
            "pixel_values": x,
            "label": target_y,
        }

    def __len__(self) -> int:
        return len(self.Y)

@dataclass
class MatchedEditedDataset:
    normal_dataset: Dataset
    qn_dataset: QuantileNormalizedDataset
    got_dataset: ConceptEditedDataset
    ce_type: Literal["got", "qn"]

    def post_init(self):
        assert len(self.normal_dataset) == len(self.qn_dataset) == len(self.ce_dataset)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if self.ce_type == "got":
            return self.normal_dataset[idx], self.got_dataset[idx]
        elif self.ce_type == "qn":
            return self.normal_dataset[idx], self.qn_dataset[idx]
        else:
            raise ValueError(f"Unknown ce_type: {self.ce_type}")
    
    def __len__(self) -> int:
        return len(self.normal_dataset)