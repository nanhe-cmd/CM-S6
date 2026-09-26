import torch

from models import (
    ABLATIONS,
    CrossModalSelectiveScan,
    FeatureMatchedSelectiveScan,
    MeanAggregationSelectiveScan,
    ModulationPaths,
    build_ablation_model,
    build_model,
    parameter_count,
)


CONFIGS = {
    "houston": {"channels": [20, 1], "num_classes": 15, "model": {"dim": 96, "state_dim": 16, "layers": 1}},
    "augsburg": {"channels": [20, 4, 1], "num_classes": 7, "model": {"dim": 96, "state_dim": 16, "layers": 1}},
    "muufl": {"channels": [20, 2], "num_classes": 11, "model": {"dim": 96, "state_dim": 16, "layers": 1}},
}


EXPECTED_PARAMETERS = {
    ("houston", "full"): 662759,
    ("houston", "lite"): 187948,
    ("augsburg", "full"): 982267,
    ("augsburg", "lite"): 190916,
    ("muufl", "full"): 663235,
    ("muufl", "lite"): 188424,
}


def test_reported_parameter_counts():
    for key, expected in EXPECTED_PARAMETERS.items():
        dataset, variant = key
        assert parameter_count(build_model(CONFIGS[dataset], variant)) == expected


def test_full_and_lite_forward_shapes():
    for dataset, config in CONFIGS.items():
        inputs = [torch.randn(1, channels, 3, 3) for channels in config["channels"]]
        for variant in ("full", "lite"):
            model = build_model(config, variant).eval()
            with torch.no_grad():
                output = model(*inputs)
            assert output.shape == (1, config["num_classes"])
            assert torch.isfinite(output).all()


def test_effective_scalars_are_tanh_bounded():
    block = CrossModalSelectiveScan(dim=8, state_dim=2, paths=ModulationPaths())
    for name in ("alpha", "beta", "gamma", "eta", "zeta"):
        getattr(block, name).data.fill_(100.0)
        assert block.effective_scalar(name).item() < 1.0
        assert block.effective_scalar(name).item() > 0.999
        getattr(block, name).data.fill_(-100.0)
        assert block.effective_scalar(name).item() > -1.0
        assert block.effective_scalar(name).item() < -0.999


def test_feature_matched_and_mean_aggregation_blocks():
    inputs = [torch.randn(1, channels, 3, 3) for channels in CONFIGS["augsburg"]["channels"]]
    model = build_ablation_model(CONFIGS["augsburg"], "A15").eval()
    with torch.no_grad():
        output = model(*inputs)
    assert output.shape == (1, CONFIGS["augsburg"]["num_classes"])
    model = build_ablation_model(CONFIGS["augsburg"], "A16").eval()
    with torch.no_grad():
        output = model(*inputs)
    assert output.shape == (1, CONFIGS["augsburg"]["num_classes"])
    assert MeanAggregationSelectiveScan is not FeatureMatchedSelectiveScan


def test_a16_is_augsburg_only():
    try:
        build_ablation_model(CONFIGS["houston"], "A16")
    except ValueError:
        return
    raise AssertionError("A16 must be rejected for two-modal datasets")


def test_all_ablation_models_build():
    for experiment in ABLATIONS:
        if experiment == "A16":
            config = CONFIGS["augsburg"]
        else:
            config = CONFIGS["houston"]
        assert parameter_count(build_ablation_model(config, experiment)) > 0
