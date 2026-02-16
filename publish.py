
import os
import sys
import subprocess
from datetime import datetime

def run_command(command, cwd=None):
    """Runs a shell command and prints the output."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e.stderr)
        return False

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Rebuild Index
    print("--- Step 1: Rebuilding Index ---")
    build_script = os.path.join(base_dir, "build_index.py")
    if not run_command(f"python3 \"{build_index_script}\"", cwd=base_dir):
        print("Failed to rebuild index. Aborting.")
        return

    # 2. Git Operations
    print("\n--- Step 2: Git Operations ---")
    
    # Check status
    if not run_command("git status", cwd=base_dir):
        return

    # Add all changes
    if not run_command("git add .", cwd=base_dir):
        print("Failed to stage files. Aborting.")
        return

    # Commit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_msg = sys.argv[1] if len(sys.argv) > 1 else f"Auto-update: {timestamp}"
    
    # Check if there are changes to commit
    status_result = subprocess.run("git status --porcelain", cwd=base_dir, shell=True, capture_output=True, text=True)
    if not status_result.stdout.strip():
        print("No changes to commit.")
    else:
        if not run_command(f"git commit -m \"{commit_msg}\"", cwd=base_dir):
            print("Failed to commit. Aborting.")
            return

    # Push
    print("\n--- Step 3: Pushing to GitHub ---")
    if run_command("git push origin main", cwd=base_dir):
        print("\nSUCCESS: Published to https://insightque.github.io/InsightFromPapers/")
    else:
        print("\nFailed to push to GitHub.")

if __name__ == "__main__":
    # Ensure build_index.py exists
    build_index_script = "build_index.py"
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), build_index_script)):
        print(f"Error: {build_index_script} not found in script directory.")
        sys.exit(1)
        
    main()
