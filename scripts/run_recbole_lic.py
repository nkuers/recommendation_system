# scripts/run_recbole_lic.py
import torch
from recbole.quick_start import run_recbole


_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat


def main():
    run_recbole(
        model="LICRec",
        dataset="ml-100k",
        config_file_list=[
            "recbole_config/ml-100k.yaml",
            "recbole_config/lic.yaml",
        ],
    )


if __name__ == "__main__":
    main()
