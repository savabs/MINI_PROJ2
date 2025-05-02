import subprocess
import os

# Directory to save the files
output_dir = 'all_versions'
os.makedirs(output_dir, exist_ok=True)

# Get all commits
commits = subprocess.check_output(['git', 'rev-list', '--all']).decode('utf-8').split()

# For each commit, get the files changed, created, or deleted
for commit in commits:
    output = subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-status', '-r', commit]).decode('utf-8').strip()
    for line in output.split('\n'):
        if line:
            status, file_path = line.split('\t')
            # Clean file path for saving
            safe_file_path = file_path.replace('/', '_')
            # Checkout the file from the commit
            subprocess.run(['git', 'checkout', commit, '--', file_path])
            # Save the file with commit hash in the name
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read()
                with open(os.path.join(output_dir, f'{safe_file_path}_{commit}'), 'wb') as f:
                    f.write(content)

print('All versions of changed files saved in', output_dir)
