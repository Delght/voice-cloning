import argparse
import os

from huggingface_hub import snapshot_download


def download_model(repo_id: str, output_dir: str):
    print(f"Downloading model {repo_id} to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=output_dir, local_dir_use_symlinks=False)
    print("Download completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download VibeVoice model")
    # Using 'microsoft/VibeVoice-Realtime-0.5B' based on the repo name.
    # The user can change it if the exact repo_id is different.
    parser.add_argument(
        "--repo_id",
        type=str,
        default="microsoft/VibeVoice-Realtime-0.5B",
        help="HuggingFace Repo ID",
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/vibevoice", help="Local output directory"
    )
    args = parser.parse_args()

    download_model(args.repo_id, args.output_dir)
