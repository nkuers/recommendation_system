# scripts/run_recbole_test_only.py
import argparse
from recbole.quick_start import load_data_and_model
from recbole.utils import get_trainer

import torch


_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat


def parse_args():
    parser = argparse.ArgumentParser(description="RecBole test-only from checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to saved model .pth")
    return parser.parse_args()


def main():
    args = parse_args()
    config, model, _, _, _, test_data = load_data_and_model(args.checkpoint)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config["show_progress"])
    print(test_result)


if __name__ == "__main__":
    main()
