from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .selective_scan import selective_scan


@dataclass(frozen=True)
class ModulationPaths:
    eta: bool = True
    alpha: bool = True
    beta: bool = True
    gamma: bool = True
    zeta: bool = True


ABLATIONS = {
    "A0": "Single-S6",
    "A1": "CatFuse-S6",
    "A2": "eta",
    "A3": "eta+alpha",
    "A4": "eta+alpha+beta",
    "A5": "eta+alpha+beta+gamma",
    "A6": "CM-S6",
    "A7": "without eta",
    "A8": "without alpha",
    "A9": "without beta",
    "A10": "without gamma",
    "A11": "without zeta",
    "A12": "CII+zeta",
    "A13": "Delta/B/C",
    "A14": "CM-S6(L)",
    "A15": "Feature-matched",
    "A16": "Mean aggregation",
}


ABLATION_PATHS = {
    "A2": ModulationPaths(True, False, False, False, False),
    "A3": ModulationPaths(True, True, False, False, False),
    "A4": ModulationPaths(True, True, True, False, False),
    "A5": ModulationPaths(True, True, True, True, False),
    "A6": ModulationPaths(),
    "A7": ModulationPaths(False, True, True, True, True),
    "A8": ModulationPaths(True, False, True, True, True),
    "A9": ModulationPaths(True, True, False, True, True),
    "A10": ModulationPaths(True, True, True, False, True),
    "A11": ModulationPaths(True, True, True, True, False),
    "A12": ModulationPaths(True, False, False, False, True),
    "A13": ModulationPaths(False, True, True, True, False),
    "A14": ModulationPaths(),
    "A15": ModulationPaths(False, True, True, True, False),
    "A16": ModulationPaths(),
}


class PatchEmbedding(nn.Module):
    def __init__(self, input_channels, dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(input_channels, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.projection(x)


class CrossModalSelectiveScan(nn.Module):
    def __init__(self, dim=96, state_dim=16, expand=2, paths=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = dim * expand
        self.state_dim = state_dim
        self.paths = paths or ModulationPaths()

        self.norm = nn.LayerNorm(dim)
        self.input_projection = nn.Linear(dim, self.hidden_dim * 2)
        self.local_conv = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=1,
            groups=self.hidden_dim,
        )
        self.delta_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        if self.paths.alpha:
            self.delta_cross_norm = nn.LayerNorm(dim)
            projection = nn.Linear(dim, self.hidden_dim, bias=False)
            nn.init.normal_(projection.weight, std=0.03)
            self.delta_cross_projection = spectral_norm(projection)
            self.delta_cross_dropout = nn.Dropout(0.2)
            self.alpha = nn.Parameter(torch.full((1,), 0.10))

        state_base = torch.arange(1, state_dim + 1).float()
        self.a_log = nn.Parameter(torch.log(state_base.unsqueeze(0).repeat(self.hidden_dim, 1)))
        self.b_projection = nn.Linear(self.hidden_dim, state_dim)
        self.c_projection = nn.Linear(self.hidden_dim, state_dim)
        self.d = nn.Parameter(torch.ones(self.hidden_dim))
        self.output_projection = nn.Linear(self.hidden_dim, dim)

        if self.paths.beta:
            self.b_cross_norm = nn.LayerNorm(dim)
            projection = nn.Linear(dim, state_dim, bias=False)
            nn.init.normal_(projection.weight, std=0.05)
            self.b_cross_projection = spectral_norm(projection)
            self.beta = nn.Parameter(torch.full((1,), 0.15))

        if self.paths.gamma:
            self.c_cross_norm = nn.LayerNorm(dim)
            projection = nn.Linear(dim, state_dim, bias=False)
            nn.init.normal_(projection.weight, std=0.05)
            self.c_cross_projection = spectral_norm(projection)
            self.gamma = nn.Parameter(torch.full((1,), 0.15))

        if self.paths.eta:
            self.input_cross_norm = nn.LayerNorm(dim)
            projection = nn.Linear(dim, self.hidden_dim, bias=False)
            nn.init.normal_(projection.weight, std=0.02)
            self.input_cross_projection = spectral_norm(projection)
            self.input_cross_dropout = nn.Dropout(0.2)
            self.eta = nn.Parameter(torch.full((1,), 0.10))

        if self.paths.zeta:
            self.output_cross_norm = nn.LayerNorm(dim)
            projection = nn.Linear(dim, dim, bias=False)
            nn.init.normal_(projection.weight, std=0.02)
            self.output_cross_projection = spectral_norm(projection)
            self.zeta = nn.Parameter(torch.full((1,), 0.05))

        self.cross_compression = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
        )
        self.gate_norm = nn.LayerNorm(dim)

    def effective_scalar(self, name):
        return getattr(self, name).tanh()

    def _aggregate_auxiliary(self, primary, auxiliary):
        if not isinstance(auxiliary, (list, tuple)):
            auxiliary = [auxiliary]

        compressed = [self.cross_compression(item.float()) for item in auxiliary]
        if len(compressed) == 1:
            return compressed[0]

        query = self.gate_norm(primary.float().mean(dim=1))
        scores = []
        for item in compressed:
            key = self.gate_norm(item.mean(dim=1))
            scores.append((query * key).sum(dim=-1, keepdim=True))
        gates = torch.softmax(torch.stack(scores, dim=1), dim=1)
        stacked = torch.stack(compressed, dim=1)
        return (stacked * gates.unsqueeze(-1)).sum(dim=1)

    def forward(self, primary, auxiliary=None):
        residual = primary
        x = self.norm(primary.float())
        cross = None
        if auxiliary is not None:
            cross = self._aggregate_auxiliary(primary, auxiliary)

        x, gate = self.input_projection(x).chunk(2, dim=-1)
        x = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            if self.paths.eta and cross is not None:
                injection = self.input_cross_projection(self.input_cross_norm(cross.float()))
                injection = torch.tanh(self.input_cross_dropout(injection))
                x = x + self.effective_scalar("eta") * injection

            delta_self = self.delta_projection(x)
            if self.paths.alpha and cross is not None:
                delta_cross = self.delta_cross_projection(self.delta_cross_norm(cross.float()))
                delta_cross = self.delta_cross_dropout(delta_cross)
                perturbation = 1.0 + self.effective_scalar("alpha") * torch.tanh(delta_cross)
                delta = F.softplus(delta_self * perturbation).clamp(max=5.0)
            else:
                delta = F.softplus(delta_self).clamp(max=5.0)

            a = -torch.exp(self.a_log.float())
            b = self.b_projection(x)
            if self.paths.beta and cross is not None:
                b_cross = self.b_cross_projection(self.b_cross_norm(cross.float()))
                b = b + self.effective_scalar("beta") * torch.tanh(b_cross)

            c = self.c_projection(x)
            if self.paths.gamma and cross is not None:
                c_cross = self.c_cross_projection(self.c_cross_norm(cross.float()))
                c = c + self.effective_scalar("gamma") * torch.tanh(c_cross)

            y = selective_scan(x, delta, a, b, c, self.d)
            y = y.clamp(-1e4, 1e4)

        y = self.output_projection(y.to(gate.dtype) * F.silu(gate))
        if self.paths.zeta and cross is not None:
            correction = self.output_cross_projection(self.output_cross_norm(cross.float()))
            y = y + self.effective_scalar("zeta") * torch.tanh(correction)
        return (y + residual).to(primary.dtype)


class FeatureMatchedSelectiveScan(CrossModalSelectiveScan):
    def forward(self, primary, auxiliary=None):
        residual = primary
        x = self.norm(primary.float())
        cross = None
        if auxiliary is not None:
            cross = self._aggregate_auxiliary(primary, auxiliary)

        x, gate = self.input_projection(x).chunk(2, dim=-1)
        x = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            x_scan = x
            if cross is not None:
                delta_cross = self.delta_cross_projection(self.delta_cross_norm(cross.float()))
                delta_cross = self.delta_cross_dropout(delta_cross)
                b_cross = self.b_cross_projection(self.b_cross_norm(cross.float()))
                c_cross = self.c_cross_projection(self.c_cross_norm(cross.float()))
                b_gate = torch.tanh(b_cross.mean(dim=-1, keepdim=True))
                c_gate = torch.tanh(c_cross.mean(dim=-1, keepdim=True))
                x_scan = (
                    x
                    + self.effective_scalar("alpha") * torch.tanh(delta_cross)
                    + self.effective_scalar("beta") * b_gate * x
                    + self.effective_scalar("gamma") * c_gate * x
                )

            delta = F.softplus(self.delta_projection(x)).clamp(max=5.0)
            a = -torch.exp(self.a_log.float())
            b = self.b_projection(x)
            c = self.c_projection(x)
            y = selective_scan(x_scan, delta, a, b, c, self.d)
            y = y.clamp(-1e4, 1e4)

        y = self.output_projection(y.to(gate.dtype) * F.silu(gate))
        return (y + residual).to(primary.dtype)


class MeanAggregationSelectiveScan(CrossModalSelectiveScan):
    def _aggregate_auxiliary(self, primary, auxiliary):
        if not isinstance(auxiliary, (list, tuple)):
            auxiliary = [auxiliary]
        compressed = [self.cross_compression(item.float()) for item in auxiliary]
        if len(compressed) == 1:
            return compressed[0]
        return torch.stack(compressed, dim=1).mean(dim=1)


def _classifier_head(dim, num_classes):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(dim, num_classes),
    )


def _to_sequence(embedding, image):
    return embedding(image).flatten(2).transpose(1, 2)


class DualModalCMS6(nn.Module):
    def __init__(self, channels, num_classes, dim=96, state_dim=16, layers=1, paths=None):
        super().__init__()
        self.primary_embedding = PatchEmbedding(channels[0], dim)
        self.auxiliary_embedding = PatchEmbedding(channels[1], dim)

        def stack():
            return nn.ModuleList(
                [CrossModalSelectiveScan(dim, state_dim, paths=paths) for _ in range(layers)]
            )

        self.primary_forward = stack()
        self.primary_backward = stack()
        self.auxiliary_forward = stack()
        self.auxiliary_backward = stack()
        self.fusion_weights = nn.Parameter(torch.ones(4))
        self.classifier = _classifier_head(dim, num_classes)

    def forward(self, primary, auxiliary):
        primary_sequence = _to_sequence(self.primary_embedding, primary)
        auxiliary_sequence = _to_sequence(self.auxiliary_embedding, auxiliary)

        primary_forward, auxiliary_forward = primary_sequence, auxiliary_sequence
        for primary_block, auxiliary_block in zip(
            self.primary_forward, self.auxiliary_forward
        ):
            primary_forward = primary_block(primary_forward, auxiliary_forward)
            auxiliary_forward = auxiliary_block(auxiliary_forward, primary_forward)

        primary_backward = primary_sequence.flip(1)
        auxiliary_backward = auxiliary_sequence.flip(1)
        for primary_block, auxiliary_block in zip(
            self.primary_backward, self.auxiliary_backward
        ):
            primary_backward = primary_block(primary_backward, auxiliary_backward)
            auxiliary_backward = auxiliary_block(auxiliary_backward, primary_backward)
        primary_backward = primary_backward.flip(1)
        auxiliary_backward = auxiliary_backward.flip(1)

        weights = torch.softmax(self.fusion_weights, dim=0)
        fused = (
            weights[0] * primary_forward
            + weights[1] * primary_backward
            + weights[2] * auxiliary_forward
            + weights[3] * auxiliary_backward
        )
        return self.classifier(fused.mean(dim=1))


class TriModalCMS6(nn.Module):
    def __init__(self, channels, num_classes, dim=96, state_dim=16, layers=1, paths=None):
        super().__init__()
        self.hsi_embedding = PatchEmbedding(channels[0], dim)
        self.sar_embedding = PatchEmbedding(channels[1], dim)
        self.dsm_embedding = PatchEmbedding(channels[2], dim)

        def stack():
            return nn.ModuleList(
                [CrossModalSelectiveScan(dim, state_dim, paths=paths) for _ in range(layers)]
            )

        self.hsi_forward = stack()
        self.hsi_backward = stack()
        self.sar_forward = stack()
        self.sar_backward = stack()
        self.dsm_forward = stack()
        self.dsm_backward = stack()
        self.fusion_weights = nn.Parameter(torch.ones(6))
        self.classifier = _classifier_head(dim, num_classes)

    @staticmethod
    def _scan(hsi, sar, dsm, hsi_blocks, sar_blocks, dsm_blocks):
        for hsi_block, sar_block, dsm_block in zip(hsi_blocks, sar_blocks, dsm_blocks):
            hsi = hsi_block(hsi, [sar, dsm])
            sar = sar_block(sar, hsi)
            dsm = dsm_block(dsm, hsi)
        return hsi, sar, dsm

    def forward(self, hsi, sar, dsm):
        hsi_sequence = _to_sequence(self.hsi_embedding, hsi)
        sar_sequence = _to_sequence(self.sar_embedding, sar)
        dsm_sequence = _to_sequence(self.dsm_embedding, dsm)

        forward = self._scan(
            hsi_sequence,
            sar_sequence,
            dsm_sequence,
            self.hsi_forward,
            self.sar_forward,
            self.dsm_forward,
        )
        backward = self._scan(
            hsi_sequence.flip(1),
            sar_sequence.flip(1),
            dsm_sequence.flip(1),
            self.hsi_backward,
            self.sar_backward,
            self.dsm_backward,
        )
        backward = tuple(item.flip(1) for item in backward)

        outputs = (forward[0], backward[0], forward[1], backward[1], forward[2], backward[2])
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused = sum(weight * output for weight, output in zip(weights, outputs))
        return self.classifier(fused.mean(dim=1))


class CMS6Lite(nn.Module):
    def __init__(self, channels, num_classes, dim=96, state_dim=16, paths=None):
        super().__init__()
        self.primary_embedding = PatchEmbedding(channels[0], dim)
        self.auxiliary_embeddings = nn.ModuleList(
            [PatchEmbedding(channel, dim) for channel in channels[1:]]
        )
        self.block = CrossModalSelectiveScan(dim, state_dim, paths=paths)
        self.classifier = _classifier_head(dim, num_classes)

    def forward(self, primary, *auxiliary):
        if len(auxiliary) != len(self.auxiliary_embeddings):
            raise ValueError(f"Expected {len(self.auxiliary_embeddings)} auxiliary inputs")
        primary_sequence = _to_sequence(self.primary_embedding, primary)
        auxiliary_sequences = [
            _to_sequence(embedding, image)
            for embedding, image in zip(self.auxiliary_embeddings, auxiliary)
        ]
        cross = auxiliary_sequences[0] if len(auxiliary_sequences) == 1 else auxiliary_sequences
        output = self.block(primary_sequence, cross)
        return self.classifier(output.mean(dim=1))


class SingleS6(nn.Module):
    def __init__(self, channels, num_classes, dim=96, state_dim=16):
        super().__init__()
        self.embedding = PatchEmbedding(channels[0], dim)
        paths = ModulationPaths(False, False, False, False, False)
        self.forward_block = CrossModalSelectiveScan(dim, state_dim, paths=paths)
        self.backward_block = CrossModalSelectiveScan(dim, state_dim, paths=paths)
        self.fusion_weights = nn.Parameter(torch.ones(2))
        self.classifier = _classifier_head(dim, num_classes)

    def forward(self, primary, *unused):
        sequence = _to_sequence(self.embedding, primary)
        forward = self.forward_block(sequence)
        backward = self.backward_block(sequence.flip(1)).flip(1)
        weights = torch.softmax(self.fusion_weights, dim=0)
        return self.classifier((weights[0] * forward + weights[1] * backward).mean(dim=1))


class CatFuseS6(nn.Module):
    def __init__(self, channels, num_classes, dim=96, state_dim=16):
        super().__init__()
        input_channels = sum(channels)
        self.input_count = len(channels)
        self.primary_embedding = PatchEmbedding(input_channels, dim)
        self.auxiliary_embedding = PatchEmbedding(input_channels, dim)
        paths = ModulationPaths(False, False, False, False, False)
        self.primary_forward = CrossModalSelectiveScan(dim, state_dim, paths=paths)
        self.primary_backward = CrossModalSelectiveScan(dim, state_dim, paths=paths)
        self.auxiliary_forward = CrossModalSelectiveScan(dim, state_dim, paths=paths)
        self.auxiliary_backward = CrossModalSelectiveScan(dim, state_dim, paths=paths)
        self.fusion_weights = nn.Parameter(torch.ones(4))
        self.classifier = _classifier_head(dim, num_classes)

    def forward(self, *modalities):
        if len(modalities) != self.input_count:
            raise ValueError(f"Expected {self.input_count} modality inputs")
        primary_image = torch.cat(modalities, dim=1)
        auxiliary_image = torch.cat(modalities[1:] + modalities[:1], dim=1)
        primary = _to_sequence(self.primary_embedding, primary_image)
        auxiliary = _to_sequence(self.auxiliary_embedding, auxiliary_image)
        outputs = (
            self.primary_forward(primary),
            self.primary_backward(primary.flip(1)).flip(1),
            self.auxiliary_forward(auxiliary),
            self.auxiliary_backward(auxiliary.flip(1)).flip(1),
        )
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused = sum(weight * output for weight, output in zip(weights, outputs))
        return self.classifier(fused.mean(dim=1))


def build_model(config, variant="full", paths=None):
    channels = tuple(config["channels"])
    model_config = config.get("model", {})
    arguments = {
        "channels": channels,
        "num_classes": config["num_classes"],
        "dim": model_config.get("dim", 96),
        "state_dim": model_config.get("state_dim", 16),
    }
    paths = paths or ModulationPaths()
    if variant == "lite":
        return CMS6Lite(**arguments, paths=paths)
    if variant != "full":
        raise ValueError(f"Unsupported CM-S6 variant: {variant}")
    arguments["layers"] = model_config.get("layers", 1)
    if len(channels) == 2:
        return DualModalCMS6(**arguments, paths=paths)
    if len(channels) == 3:
        return TriModalCMS6(**arguments, paths=paths)
    raise ValueError("CM-S6 expects two or three modalities")


def build_ablation_model(config, experiment):
    experiment = experiment.upper()
    if experiment not in ABLATIONS:
        raise ValueError(f"Unknown ablation experiment: {experiment}")
    channels = tuple(config["channels"])
    model_config = config.get("model", {})
    arguments = {
        "channels": channels,
        "num_classes": config["num_classes"],
        "dim": model_config.get("dim", 96),
        "state_dim": model_config.get("state_dim", 16),
    }
    if experiment == "A0":
        return SingleS6(**arguments)
    if experiment == "A1":
        return CatFuseS6(**arguments)
    if experiment == "A14":
        return CMS6Lite(**arguments, paths=ABLATION_PATHS[experiment])
    if experiment in ("A15", "A16"):
        if experiment == "A16" and len(channels) != 3:
            raise ValueError("A16 mean aggregation is defined for the three-modal Augsburg setting")
        block_class = FeatureMatchedSelectiveScan if experiment == "A15" else MeanAggregationSelectiveScan
        original = globals()["CrossModalSelectiveScan"]
        globals()["CrossModalSelectiveScan"] = block_class
        try:
            return build_model(config, variant="full", paths=ABLATION_PATHS[experiment])
        finally:
            globals()["CrossModalSelectiveScan"] = original
    return build_model(config, variant="full", paths=ABLATION_PATHS[experiment])


def modulation_blocks(model):
    return [module for module in model.modules() if isinstance(module, CrossModalSelectiveScan)]


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())
