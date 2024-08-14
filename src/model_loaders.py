import os
import torch
from safetensors.torch import load_file

from transformers import AutoModelForCausalLM, ConvNextV2ForImageClassification
from torchvision.models import regnet_y_1_6gf, regnet_y_3_2gf, regnet_y_400mf, regnet_y_800mf
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformer, SwinTransformerBlockV2


def load_pythia(model_name: str, revision: str):
    return AutoModelForCausalLM.from_pretrained(model_name, revision=revision)


class HfWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values)


def load_image_model(model_path: str, model_arch: str, num_classes: int):
    match model_arch.partition("-"):
        case ("convnext", _, arch):
            model = ConvNextV2ForImageClassification.from_pretrained(model_path).cuda() # type: ignore
        case ("regnet", _, arch):
            match arch:
                case "400mf":
                    net = regnet_y_400mf(num_classes=num_classes)
                case "800mf":
                    net = regnet_y_800mf(num_classes=num_classes)
                case "1.6gf":
                    net = regnet_y_1_6gf(num_classes=num_classes)
                case "3.2gf":
                    net = regnet_y_3_2gf(num_classes=num_classes)
                case other:
                    raise ValueError(f"Unknown RegNet architecture {other}")

            net.stem[0].stride = (1, 1) # type: ignore
            model = HfWrapper(net)
            state_dict_path = os.path.join(model_path, "model.safetensors")
            model.load_state_dict(load_file(state_dict_path))
            model.cuda()
        case ("swin", _, arch):
            match arch:
                case "atto":
                    num_heads = [2, 4, 8, 16]
                    embed_dim = 40
                case "femto":
                    num_heads = [2, 4, 8, 16]
                    embed_dim = 48
                case "pico":
                    num_heads = [2, 4, 8, 16]
                    embed_dim = 64
                case "nano":
                    num_heads = [2, 4, 8, 16]
                    embed_dim = 80
                case "tiny" | "":  # default
                    num_heads = [3, 6, 12, 24]
                    embed_dim = 96
                case other:
                    raise ValueError(f"Unknown Swin architecture {other}")

            swin = SwinTransformer(
                patch_size=[2, 2],
                embed_dim=embed_dim,
                depths=[2, 2, 6, 2],
                num_heads=num_heads,
                window_size=[7, 7],
                num_classes=num_classes,
                stochastic_depth_prob=0.2,
                block=SwinTransformerBlockV2,
                downsample_layer=PatchMergingV2,
            )
            model = HfWrapper(swin)
            state_dict_path = os.path.join(model_path, "model.safetensors")
            model.load_state_dict(load_file(state_dict_path))
            model.cuda()
        case _:
            raise ValueError(f"Unknown model {model_arch}")

    return model