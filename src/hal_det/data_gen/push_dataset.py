import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# Load environment variables (HF_TOKEN)
load_dotenv()

# Configuration
HF_USERNAME = "pagand"
DATASET_NAME = "venra" 
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"
DATA_DIR = Path("data/training_final")
HF_TOKEN = os.getenv("HF_TOKEN")

def upload_to_huggingface(version_tag: str = "v1.0"):
    if not HF_TOKEN:
        print("❌ Error: HF_TOKEN not found in .env file.")
        print("Please add HF_TOKEN=hf_... to your .env file.")
        return

    print(f"Preparing to upload to {REPO_ID} (Tag: {version_tag})...")
    
    # Initialize API with token
    api = HfApi(token=HF_TOKEN)
    
    # 1. Create the Repo (if it doesn't exist)
    try:
        create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN, private=True)
        print(f"✅ Repository {REPO_ID} is ready.")
    except Exception as e:
        print(f"❌ Error creating repo: {e}")
        return

    # 2. Upload Files
    files = ["train.jsonl", "val.jsonl", "test.jsonl", "README.md"]
    
    for filename in files:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            print(f"⚠️ Warning: {filename} not found at {file_path}")
            continue
            
        print(f"Uploading {filename}...")
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Upload {filename} (VeNRA Pipeline Output - {version_tag})"
            )
            print(f"✅ Uploaded {filename}")
        except Exception as e:
            print(f"❌ Failed to upload {filename}: {e}")

    # 3. Create a Git Tag for Versioning
    try:
        api.create_tag(
            repo_id=REPO_ID, 
            tag=version_tag, 
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"✅ Tag {version_tag} created.")
    except Exception as e:
        # If tag exists, just warn
        if "already exists" in str(e).lower():
            print(f"ℹ️ Note: Tag {version_tag} already exists. Skipping tag creation.")
        else:
            print(f"⚠️ Warning: Failed to create tag: {e}")

    print("\n🎉 Success! View your dataset here:")
    print(f"https://huggingface.co/datasets/{REPO_ID}")
    print(f"To use this version in code: load_dataset('{REPO_ID}', revision='{version_tag}')")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push VeNRA dataset to Hugging Face Hub.")
    parser.add_argument("--tag", type=str, default="v1.0", help="Version tag for this upload (e.g. v1.1)")
    args = parser.parse_args()
    
    upload_to_huggingface(version_tag=args.tag)
