#!/bin/bash

# SSH Key Generation
echo -e "\n"
read -p "Enter your GitHub email: " email
read -p "Enter your GitHub username: " username
key_file="$HOME/.ssh/id_ed25519_${username}"

echo -e "\n\n===== Generating SSH key for GitHub =====\n\n"
ssh-keygen -t ed25519 -C "$email" -f "$key_file"

# Start the SSH agent
echo -e "\n\n===== Starting SSH agent =====\n\n"
eval "$(ssh-agent -s)"

# Add SSH private key to the agent
echo -e "\n\n===== Adding SSH key to SSH agent =====\n\n"
ssh-add "$key_file"

# Configure SSH for GitHub
echo -e "\n\n===== Configuring SSH for GitHub =====\n\n"
ssh_config="$HOME/.ssh/config"
if ! grep -q "Host github.com" "$ssh_config"; then
    echo -e "\nHost github.com\n  HostName github.com\n  User git\n  IdentityFile $key_file\n" >> "$ssh_config"
    chmod 600 "$ssh_config"
fi

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
repo_name=$(basename "$repo" .git)
mkdir $username
cd "$username" || exit
git clone "$repo"

# Configure Git for this repository
echo -e "\n\n===== Configuring Git user details for this repository =====\n\n"
git config --local user.email "$email"
git config --local user.name "$username"