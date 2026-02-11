import importlib


def resolve_optimizer_cls(optimizer_cfg: dict) -> tuple[type, dict]:
    """
    Pops _target_ from cfg and returns (optimizer_class, remaining_kwargs).
    Assumes configs for the optimizer are structued likle so:
        ```
        _target_: torch.optim.Adam
        _partial_: true
        lr: 1e-3
        ```
    """
    optimizer_cfg = optimizer_cfg.copy()
    target = optimizer_cfg.pop("_target_", False)
    optimizer_cfg.pop("_partial_", False)
    module_path, cls_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls, optimizer_cfg
