

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Update pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Handle specific installation steps for maturin dependencies if needed
pip install maturin

# Install your Python package
pip install .
