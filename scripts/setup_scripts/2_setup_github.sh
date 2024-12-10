#!/bin/bash

# SSH Key Generation
echo -e "\n"
read -p "Enter your GitHub email: " email
read -p "Enter your GitHub username: " username
key_file="$HOME/.ssh/id_ed25519_${username}"
ssh_config="$HOME/.ssh/config"

# Check if the .ssh file exists
if [[ ! -f $key_file ]]; then
    echo -e "\n\n===== Generating SSH key for GitHub =====\n\n"
    ssh-keygen -t ed25519 -C "$email" -f "$key_file"
else
    echo -e "\n===== $key_file already exists. Skipping creation =====\n"
fi

# Ensure the .ssh directory exists
if [[ ! -d "$HOME/.ssh" ]]; then
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
else
    echo -e "\n===== $HOME/.ssh already exists. Skipping creation =====\n"
fi

# Ensure the config file exists
if [[ ! -f "$ssh_config" ]]; then
    echo -e "\n===== Creating SSH config file =====\n"
    touch "$ssh_config"
    chmod 600 "$ssh_config"
else
    echo -e "\n===== $ssh_config already exists. Skipping creation =====\n"
fi

# Add GitHub configuration to the SSH config file
if ! grep -q "Host github.com-$username" "$ssh_config"; then
    echo -e "\nHost github.com-$username" >> "$ssh_config"
    echo -e "  HostName github.com" >> "$ssh_config"
    echo -e "  User git" >> "$ssh_config"
    echo -e "  IdentityFile $key_file" >> "$ssh_config"
    echo -e "  IdentitiesOnly yes\n" >> "$ssh_config"
    chmod 600 "$ssh_config"
fi

# Start the SSH agent
echo -e "\n\n===== Starting SSH agent =====\n\n"
eval "$(ssh-agent -s)"

# Add SSH private key to the agent
echo -e "\n\n===== Adding SSH key to SSH agent =====\n\n"
ssh-add "$key_file"

# Display the SSH public key
echo -e "\n\n===== Please add the SSH public key to your GitHub account before proceeding =====\n\n"
cat "${key_file}.pub"
echo -e "\n"
read -p "Press Enter once you have added the key to GitHub..."

# Clone GitHub repository
echo -e "\n\n===== Cloning the specified GitHub repository =====\n\n"
default_repo="git@github.com:ChefSteveP/Aura.git"
read -p "Enter the GitHub repository URL to clone (default: $default_repo): " repo
repo=${repo:-$default_repo}
repo_path=$(echo "$repo" | sed 's/git@github.com://')
repo_name=$(basename "$repo" .git)

# Check if the directory for the user exists; create it if it doesn't
if [[ ! -d "$username" ]]; then
    echo -e "\n===== Creating directory for user $username =====\n"
    mkdir "$username"
else
    echo -e "\n===== Directory $username already exists. Skipping creation. =====\n"
fi

# Check if the repo exists in <username>/<repo_name>; create it if it doesn't
cd "$username" || exit
if [[ ! -d "$repo_name" ]]; then
    echo -e "\n===== Cloning $repo =====\n"
    git clone "$repo"
else
    echo -e "\n===== $repo_name already exists. Skipping creation. =====\n"
fi

cd "$repo_name" || exit

# Configure Git for this repository
echo -e "\n\n===== Configuring Git user details for this repository =====\n\n"
git config --local user.email "$email"
git config --local user.name "$username"

# Final authentication
git_url="git@github.com-$username"
ssh -T $git_url
git remote set-url origin "$git_url:$repo_path"