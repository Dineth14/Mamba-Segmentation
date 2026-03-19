# MambaVision Core Package

This folder contains the core MambaVision model package code used by downstream tasks in this repository.\n\n## Purpose\n- define model architectures and building blocks\n- provide train/eval entry points used by task-specific wrappers\n- keep upstream MambaVision code locally vendored for reproducible experiments\n\n## Notes\nThis directory is a vendored dependency inside a model subtree. Prefer editing the task-level configs and wrappers at the parent module unless you are intentionally modifying upstream MambaVision internals.
