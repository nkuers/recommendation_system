# scripts/register_recbole_model.py
import argparse
import shutil
import sys
import sysconfig
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Register a custom RecBole model.")
    parser.add_argument("--model_file", required=True, help="Path to model python file")
    parser.add_argument("--model_name", required=True, help="Class name to import")
    parser.add_argument("--subdir", default="sequential_recommender", help="RecBole model subdir")
    return parser.parse_args()


def main():
    args = parse_args()
    model_file = Path(args.model_file).resolve()
    if not model_file.exists():
        raise SystemExit(f"model file not found: {model_file}")

    site_pkg = Path(sysconfig.get_paths()["purelib"]).resolve()
    target_dir = site_pkg / "recbole" / "model" / args.subdir
    init_file = target_dir / "__init__.py"
    if not target_dir.exists():
        raise SystemExit(f"target dir not found: {target_dir}")

    dst = target_dir / model_file.name
    shutil.copyfile(model_file, dst)
    print(f"[INFO] copied to {dst}")

    extra_files = []
    if model_file.stem == "timeweaver":
        time_modules = model_file.parent / "time_modules.py"
        if time_modules.exists():
            extra_files.append(time_modules)
    if model_file.stem == "tisasrec":
        ti_transformer = model_file.parent / "ti_transformer.py"
        if ti_transformer.exists():
            extra_files.append(ti_transformer)
    for extra in extra_files:
        extra_dst = target_dir / extra.name
        shutil.copyfile(extra, extra_dst)
        print(f"[INFO] copied to {extra_dst}")

    import_line = f"from recbole.model.{args.subdir}.{model_file.stem} import {args.model_name}"
    if init_file.exists():
        content = init_file.read_text(encoding="utf-8")
    else:
        content = ""
    if import_line not in content:
        with open(init_file, "a", encoding="utf-8") as f:
            if content and not content.endswith("\n"):
                f.write("\n")
            f.write(import_line + "\n")
        print(f"[INFO] updated {init_file}")
    else:
        print(f"[INFO] import already exists in {init_file}")


if __name__ == "__main__":
    main()
