"""Reproducible experiments that sit on top of the whole stack (data, privacy, split learning, scheduling).

Nothing here is imported by the coordinator, the engine or the data/privacy packages; experiments depend
on them, never the other way round. `python -m experiments.noise_comparison` runs the activation-noise
comparison described in docs/features/data-and-privacy.md.
"""
