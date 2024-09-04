from dataclasses import dataclass, field
from typing import List, Literal, Callable, Optional

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
    target_Y: Tensor  # New field for pre-generated target classes

    def __init__(self, class_probs: Tensor, editor: QuadraticEditor, X: Tensor, Y: Tensor, seed: Optional[int] = None, target_Y: Optional[Tensor] = None):
        self.class_probs = class_probs
        self.editor = editor
        self.X = X
        self.Y = Y
        self.target_Y = target_Y if target_Y is not None else self._generate_target_classes(seed)

    def _generate_target_classes(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        
        target_Y = []
        for y in self.Y:
            loo_probs = self.class_probs.clone()
            loo_probs[y] = 0
            target_y = torch.multinomial(loo_probs, 1).squeeze()
            target_Y.append(target_y)
        
        return torch.tensor(target_Y)

    def select(self, idx: List[int]):
        return ConceptEditedDataset(
            class_probs=self.class_probs,
            editor=self.editor,
            X=self.X[idx],
            Y=self.Y[idx],
            target_Y=self.target_Y[idx]
        )

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x, y = self.X[idx], int(self.Y[idx])
        target_y = int(self.target_Y[idx])

        x = self.editor.transport(x[None], y, target_y).squeeze(0)
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
    target_Y: Tensor  # New field for pre-generated target classes

    def __init__(self, class_probs: Tensor, editor: QuantileNormalizer, X: Tensor, Y: Tensor, seed: Optional[int] = None, target_Y: Optional[Tensor] = None):
        self.class_probs = class_probs
        self.editor = editor
        self.X = X
        self.Y = Y
        self.target_Y = target_Y if target_Y is not None else self._generate_target_classes(seed)

    def _generate_target_classes(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        
        target_Y = []
        for y in self.Y:
            loo_probs = self.class_probs.clone()
            loo_probs[y] = 0
            target_y = torch.multinomial(loo_probs, 1).squeeze()
            target_Y.append(target_y)
        
        return torch.tensor(target_Y)

    def select(self, idx: List[int]):
        return QuantileNormalizedDataset(
            class_probs=self.class_probs,
            editor=self.editor,
            X=self.X[idx],
            Y=self.Y[idx],
            target_Y=self.target_Y[idx]
        )

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x, y = self.X[idx], self.Y[idx]
        target_y = self.target_Y[idx]

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
class IndependentCoordinateSampler:
    class_probs: Tensor
    editor: QuantileNormalizer
    size: int

    def select(self, idx: List[int]):
        return IndependentCoordinateSampler(
            class_probs=self.class_probs,
            editor=self.editor,
            size=len(idx)
        )

    def __getitem__(self, _: int) -> dict[str, Tensor]:
        y = torch.multinomial(self.class_probs, 1).squeeze()
        lut = self.editor.lut[y]

        indices = torch.randint(0, lut.shape[-1], lut[..., 0].shape, device=lut.device)
        x = lut.gather(-1, indices[..., None]).squeeze(-1)

        return {
            "pixel_values": x,
            "label": y,
        }

    def __len__(self) -> int:
        return self.size

class GaussianMixture:
    def __init__(
        self,
        class_probs: Tensor,
        size: int,
        means: Optional[Tensor] = None,
        covs: Optional[Tensor] = None,
        dists: Optional[List[MultivariateNormal]] = None,
        shape: tuple[int, int, int] = (3, 32, 32),
        trf: Callable = lambda x: x,
    ):
        assert means is not None or dists is not None
        self.class_probs = class_probs
        if means is not None:
            self.dists = [MultivariateNormal(mean, cov) for mean, cov in zip(means, covs)]
        else:
            self.dists = dists
        self.shape = shape
        self.size = size
        self.trf = trf

    def select(self, idx: List[int]):
        return GaussianMixture(
            class_probs=self.class_probs,
            dists=self.dists,
            shape=self.shape,
            size=len(idx)
        )

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of bounds for size {self.size}")

        y = torch.multinomial(self.class_probs, 1).squeeze()
        x = self.dists[y].sample().reshape(self.shape)
        return {
            "pixel_values": self.trf(x),
            "label": y,
        }

    def __len__(self) -> int:
        return self.size


@dataclass
class MatchedEditedDataset:
    normal_dataset: Dataset
    qn_dataset: QuantileNormalizedDataset
    got_dataset: ConceptEditedDataset
    ics_dataset: IndependentCoordinateSampler
    gauss_dataset: GaussianMixture
    return_type: Literal["edited", "synthetic"]

    def post_init(self):
        assert len(self.normal_dataset) == len(self.qn_dataset) == len(self.ce_dataset)

    def select(self, idx: List[int]):
        return MatchedEditedDataset(
            normal_dataset=self.normal_dataset.select(idx),
            qn_dataset=self.qn_dataset.select(idx),
            got_dataset=self.got_dataset.select(idx),
            return_type=self.return_type,
            ics_dataset=self.ics_dataset.select(idx),
            gauss_dataset=self.gauss_dataset.select(idx)
        )

    def __getitem__(self, idx: int):
        if self.return_type == "edited":
            return self.normal_dataset[idx], self.qn_dataset[idx], self.got_dataset[idx]
        elif self.return_type == "synthetic":
            return self.normal_dataset[idx], self.ics_dataset[idx], self.gauss_dataset[idx]
        else:
            raise ValueError(f"Unknown return type: {self.return_type}")
    
    def __len__(self) -> int:
        return len(self.normal_dataset)