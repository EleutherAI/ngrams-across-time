from __future__ import annotations
import torch as t
from torchtyping import TensorType


class DenseAct():
    def __init__(
            self, 
            act: TensorType["batch_size", "n_ctx", "d_model"] = None,
            res: TensorType["batch_size", "n_ctx", "d_model"] = None,
            resc: TensorType["batch_size", "n_ctx"] = None, # contracted residual
            ) -> None:

            self.act = act
            self.res = res
            self.resc = resc

    def _map(self, f, aux=None) -> 'DenseAct':
        kwargs = {}
        if isinstance(aux, DenseAct):
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None and getattr(aux, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), getattr(aux, attr))
        else:
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), aux)
        return DenseAct(**kwargs)
        
    def __mul__(self, other) -> 'DenseAct':
        kwargs = {}

        if isinstance(other, DenseAct):
            # Handle DenseAct * DenseAct
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * getattr(other, attr)
        else:
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * other
        return DenseAct(**kwargs)

    def __rmul__(self, other) -> 'DenseAct':
        # This will handle float/int * DenseAct by reusing the __mul__ logic
        return self.__mul__(other)
    
    def __matmul__(self, other: DenseAct) -> 'DenseAct':
        return DenseAct(act = self.act * other.act, resc=(self.res * other.res).sum(dim=-1, keepdim=True))
    
    def __add__(self, other) -> 'DenseAct':
        if isinstance(other, DenseAct):
            kwargs = {}            
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    # if getattr(self, attr).shape != getattr(other, attr).shape:
                        # raise ValueError(f"Shapes of {attr} do not match: {getattr(self, attr).shape} and {getattr(other, attr).shape}")
                    kwargs[attr] = getattr(self, attr) + getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) + other
        return DenseAct(**kwargs)
    
    def __radd__(self, other: DenseAct) -> 'DenseAct':
        return self.__add__(other)
    
    def __sub__(self, other) -> 'DenseAct':
        if isinstance(other, DenseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    if getattr(self, attr).shape != getattr(other, attr).shape:
                        raise ValueError(f"Shapes of {attr} do not match: {getattr(self, attr).shape} and {getattr(other, attr).shape}")
                    kwargs[attr] = getattr(self, attr) - getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) - other
        return DenseAct(**kwargs)
    
    def __truediv__(self, other) -> 'DenseAct':
        if isinstance(other, DenseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / other
        return DenseAct(**kwargs)

    def __rtruediv__(self, other) -> 'DenseAct':
        if isinstance(other, DenseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        return DenseAct(**kwargs)

    def __neg__(self) -> 'DenseAct':
        return DenseAct(act=-self.act, res=-self.res)
    
    def __invert__(self) -> 'DenseAct':
            return self._map(lambda x, _: ~x)


    def __gt__(self, other) -> 'DenseAct':
        if isinstance(other, (int, float)):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) > other
            return DenseAct(**kwargs)
        raise ValueError("DenseAct can only be compared to a scalar.")
    
    def __lt__(self, other) -> 'DenseAct':
        if isinstance(other, (int, float)):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) < other
            return DenseAct(**kwargs)
        raise ValueError("DenseAct can only be compared to a scalar.")
    
    def __getitem__(self, index: int):
        return self.act[index]
    
    def __repr__(self):
        if self.res is None:
            return f"DenseAct(act={self.act}, resc={self.resc})"
        if self.resc is None:
            return f"DenseAct(act={self.act}, res={self.res})"
        else:
            raise ValueError("DenseAct has both residual and contracted residual. This is an unsupported state.")
    
    def sum(self, dim=None):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).sum(dim)
        return DenseAct(**kwargs)
    
    def mean(self, dim: int):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).mean(dim)
        return DenseAct(**kwargs)
    
    def nonzero(self):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).nonzero()
        return DenseAct(**kwargs)
    
    def squeeze(self, dim: int):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).squeeze(dim)
        return DenseAct(**kwargs)

    @property
    def grad(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).grad
        return DenseAct(**kwargs)
    
    def clone(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).clone()
        return DenseAct(**kwargs)
    
    @property
    def value(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).value
        return DenseAct(**kwargs)

    def save(self):
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                setattr(self, attribute, getattr(self, attribute).save())
        return self

    def save_grad(self):
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                grad = getattr(getattr(self, attribute), 'grad')
                setattr(getattr(self, attribute), 'grad', grad.save())
        return self
    
    def detach(self):
        self.act = self.act.detach()
        self.res = self.res.detach()
        return DenseAct(act=self.act, res=self.res)
    
    def to_tensor(self):
        if self.resc is None:
            return t.cat([self.act, self.res], dim=-1)
        if self.res is None:
            return t.cat([self.act, self.resc], dim=-1)
        raise ValueError("DenseAct has both residual and contracted residual. This is an unsupported state.")

    def to(self, device):
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).to(device))
        return self
    
    def __gt__(self, other):
        return self._map(lambda x, y: x > y, other)
    
    def __lt__(self, other):
        return self._map(lambda x, y: x < y, other)
    
    def nonzero(self):
        return self._map(lambda x, _: x.nonzero())
    
    def squeeze(self, dim):
        return self._map(lambda x, _: x.squeeze(dim=dim))
    
    def expand_as(self, other):
        return self._map(lambda x, y: x.expand_as(y), other)
    
    def zeros_like(self):
        return self._map(lambda x, _: t.zeros_like(x))
    
    def ones_like(self):
        return self._map(lambda x, _: t.ones_like(x))
    
    def abs(self):
        return self._map(lambda x, _: x.abs())


def to_dense(top_acts, top_indices, num_latents: int):
    dense_empty = t.zeros(top_acts.shape[0], top_acts.shape[1], num_latents, device=top_acts.device, requires_grad=True)
    return dense_empty.scatter(-1, top_indices.long(), top_acts)