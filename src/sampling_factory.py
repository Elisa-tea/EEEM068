from sampling import EquidistantSampling, InterpolationSampling, SamplingWithAug
def create_sampler(strategy, **kwargs):
    """
    Factory function to create a sampling strategy object.

    Args:
        strategy (str): Name of the sampling strategy.
        **kwargs: Arguments to be passed to the sampler class.

    Returns:
        Sampling: An instance of the selected sampling strategy.
    """
    strategy = strategy.lower()
    if strategy == "equidistant":
        return EquidistantSampling(**kwargs)
    elif strategy == "interpolation":
        return InterpolationSampling(**kwargs)
    elif strategy == "augmented":
        return SamplingWithAug(**kwargs)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")