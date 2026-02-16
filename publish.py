import os
import sys
import subprocess
from datetime import datetime

def run_command(cmd, cwd=None):
    print(f"Running: {cmd}")
    # We want to capture output to see it in the main.py capture if needed
    result = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Publishing from base_dir: {base_dir}")
    
    # Ensure build_index.py exists
    build_index_script = os.path.join(base_dir, "build_index.py")
    if not os.path.exists(build_index_script):
        print(f"Error: {build_index_script} not found.")
        sys.exit(1)

    # 1. Rebuild Index
    print("\n--- Step 1: Rebuilding Index ---")
    if not run_command(f"python3 \"{build_index_script}\"", cwd=base_dir):
        print("Failed to rebuild index. Aborting.")
        sys.exit(1)

    # 2. Git Operations
    print("\n--- Step 2: Git Operations ---")
    
    # Configure git for environment LOCAL to the repo
    # This prevents permission errors with ~/.config/gcloud
    print("Configuring local git identity...")
    run_command("git config user.email \"agent@insightque.com\"", cwd=base_dir)
    run_command("git config user.name \"Insightque Agent\"", cwd=base_dir)

    # Check status
    print("Checking git status...")
    run_command("git status", cwd=base_dir)

    # Add all changes (Modified and Untracked)
    print("Staging changes...")
    if not run_command("git add -A", cwd=base_dir):
        print("Failed to stage files. Aborting.")
        sys.exit(1)

    # Commit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_msg = sys.argv[1] if len(sys.argv) > 1 else f"Auto-update: {timestamp}"
    
    # Check if there are changes to commit
    print("Checking for changes to commit...")
    status_result = subprocess.run("git status --porcelain", cwd=base_dir, shell=True, capture_output=True, text=True)
    if not status_result.stdout.strip():
        print("No changes to commit. Proceeding to push (might be empty/already committed).")
    else:
        if not run_command(f"git commit -m \"{commit_msg}\"", cwd=base_dir):
            print("Failed to commit. Aborting.")
            sys.exit(1)

    # Push
    print("\n--- Step 3: Pushing to GitHub ---")
    
    pat = os.environ.get("GITHUB_PAT")
    if pat:
        print("Using GitHub PAT for authentication.")
        # Sanitize PAT for logging
        repo_url = f"https://{pat}@github.com/Insightque/InsightFromPapers.git"
        # Temporarily change remote URL for push
        run_command(f"git remote set-url origin {repo_url}", cwd=base_dir)
    else:
        print("Warning: GITHUB_PAT not found in environment.")
    
    if run_command("git push origin main", cwd=base_dir):
        print("\nSUCCESS: Published to https://insightque.github.io/InsightFromPapers/")
    else:
        print("\nFailed to push to GitHub.")
        sys.exit(1)

if __name__ == "__main__":
    main()
