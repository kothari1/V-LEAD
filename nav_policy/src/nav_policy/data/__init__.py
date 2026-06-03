def __getattr__(name):
    if name == "RGBHorizonDataset":
        from nav_policy.data.rgb_horizon_dataset import RGBHorizonDataset
        return RGBHorizonDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["RGBHorizonDataset"]
