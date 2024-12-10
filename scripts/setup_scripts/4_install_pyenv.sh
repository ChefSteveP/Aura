# Install pyenv dependencies
echo -e "\n\n===== Installing pyenv dependencies =====\n\n"
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install pyenv
echo -e "\n\n===== Installing pyenv =====\n\n"
curl https://pyenv.run | bash

# Add pyenv configuration to .bashrc
echo -e "\n\n===== Configuring pyenv in .bashrc =====\n\n"
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Final instructions
echo -e "\n\n===== Restart your shell or run 'exec \$SHELL' to apply changes, and then continue with additional setup. =====\n\n"
exec "$SHELL"
