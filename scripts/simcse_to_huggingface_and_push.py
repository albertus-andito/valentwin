"""
Convert SimCSE's checkpoints to Huggingface style.
"""

import argparse
import torch
import os
import json
import shutil

from huggingface_hub import create_repo, HfApi, repo_exists
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path of SimCSE checkpoint folder")
    parser.add_argument("--push", action="store_true", help="Push to Huggingface")
    parser.add_argument("--hf_repo_id", type=str, help="Huggingface repo ID to push to")
    parser.add_argument("--delete", action="store_true", help="Delete local checkpoint after pushing to Huggingface")

    args = parser.parse_args()

    print("SimCSE checkpoint -> Huggingface checkpoint for {}".format(args.path))

    path_to_pt_tensor = os.path.join(args.path, "pytorch_model.bin")
    path_to_safetensors = os.path.join(args.path, "model.safetensors")

    if os.path.exists(path_to_pt_tensor):
        state_dict = torch.load(path_to_pt_tensor, map_location=torch.device("cpu"))
    else:
        state_dict = load_file(path_to_safetensors)

    new_state_dict = {}
    for key, param in state_dict.items():
        # Replace "mlp" to "pooler"
        if "mlp" in key:
            key = key.replace("mlp", "pooler")

        # Delete "bert" or "roberta" prefix
        if "bert." in key:
            key = key.replace("bert.", "")
        if "roberta." in key:
            key = key.replace("roberta.", "")

        new_state_dict[key] = param

    if os.path.exists(path_to_pt_tensor):
        torch.save(new_state_dict, path_to_pt_tensor)
    else:
        save_file(new_state_dict, path_to_safetensors, metadata={'format': 'pt'})

    # Change architectures in config.json
    config = json.load(open(os.path.join(args.path, "config.json")))
    for i in range(len(config["architectures"])):
        config["architectures"][i] = config["architectures"][i].replace("ForCL", "Model")
    json.dump(config, open(os.path.join(args.path, "config.json"), "w"), indent=2)

    if args.push:
        # trainer.push_to_hub()
        repo_id = f"{args.hf_repo_id}/{os.path.basename(os.path.normpath(args.path))}"
        if not repo_exists(repo_id):
            create_repo(repo_id, private=True)
        api = HfApi()
        api.upload_folder(folder_path=args.path, repo_id=repo_id)
        # model = AutoModel.from_pretrained(args.path)
        # model.push_to_hub(os.path.basename(os.path.normpath(args.path)))

    if args.delete:
        shutil.rmtree(args.path)


if __name__ == "__main__":
    main()
