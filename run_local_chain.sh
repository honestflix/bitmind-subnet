#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if conda is installed
conda_exists() {
    conda --version >/dev/null 2>&1
}

# Update and install necessary packages
echo "Updating system packages..."
sudo apt update

# Install required libraries and tools if not already installed
echo "Checking and installing required libraries and tools..."
declare -a required_packages=("make" "build-essential" "git" "clang" "curl" "libssl-dev" "llvm" "libudev-dev" "protobuf-compiler")
for package in "${required_packages[@]}"; do
    if ! dpkg -l | grep -qw "$package"; then
        echo "Installing $package..."
        sudo apt install --assume-yes "$package"
    else
        echo "$package is already installed."
    fi
done

# Install Rust and Cargo if not already installed
if ! command_exists rustc; then
    echo "Installing Rust and Cargo..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source "$HOME/.cargo/env"
else
    echo "Rust and Cargo are already installed."
fi

# Check if Miniconda is installed
if ! conda_exists; then
    echo "Miniconda is not installed. Installing Miniconda..."

    # Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh

    # Install Miniconda
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

    # Initialize Miniconda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init

    # Clean up installer
    rm Miniconda3-latest-Linux-x86_64.sh

    echo "Miniconda installation complete."
else
    echo "Miniconda is already installed."
    eval "$(conda shell.bash hook)"
fi

# Create and activate the 'subnet' environment
if conda info --envs | grep -qw 'subnet'; then
    echo "Conda environment 'subnet' already exists. Activating it..."
else
    echo "Creating Conda environment 'subnet'..."
    conda create --name subnet -y
fi

echo "Activating Conda environment 'subnet'..."
conda activate subnet

echo "Conda environment 'subnet' is activated."

# Clone the subtensor repository if not already cloned
if [ ! -d "subtensor" ]; then
    echo "Cloning the subtensor repository..."
    git clone https://github.com/opentensor/subtensor.git
else
    echo "Subtensor repository already cloned."
fi



# Setup Rust for Subtensor
echo "Setting up Rust for Subtensor..."
cd subtensor
./scripts/init.sh

# Build the Subtensor binary with the faucet feature enabled
echo "Building the Subtensor binary with the faucet feature enabled..."
cargo build --release --features pow-faucet

# Install npm and pm2 if not already installed
if ! command_exists npm; then
    echo "Installing npm..."
    sudo apt install --assume-yes npm
else
    echo "npm is already installed."
fi

if ! command_exists pm2; then
    echo "Installing pm2..."
    sudo npm install -g pm2
else
    echo "pm2 is already installed."
fi

# Run the localnet script with pm2
echo "Running the localnet script with pm2..."
pm2 start ./scripts/localnet.sh --name localnet


# Wait for the subnet to get started
echo "Waiting for the local subnet to get started..."
sleep 300
echo "Subnet is now started."



# Install subnet template if not already cloned
cd ..
if [ ! -d "bittensor-subnet-template" ]; then
    echo "Cloning the bittensor subnet template repository..."
    git clone https://github.com/opentensor/bittensor-subnet-template.git
else
    echo "Bittensor subnet template repository already cloned."
fi

# Navigate to the cloned repository
cd bittensor-subnet-template
echo "Installing the bittensor-subnet-template Python package..."
python -m pip install -e .

# Create wallets for owner, miner, and validator
echo "Creating wallets for owner, miner, and validator..."
btcli wallet new_coldkey --wallet.name owner
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default

# Mint tokens from faucet for owner and validator
echo "Minting tokens from faucet..."
btcli wallet faucet --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli wallet faucet --wallet.name validator --subtensor.chain_endpoint ws://127.0.0.1:9946

# Create a subnet
echo "Creating a subnet..."
btcli subnet create --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946

# Register subnet miner and validator
echo "Registering subnet miner and validator..."
btcli subnet register --wallet.name miner --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli subnet register --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946

# Add stake for the validator
echo "Adding stake for the validator..."
btcli stake add --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946

# Verify key registrations
echo "Verifying key registrations..."
btcli subnet list --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli wallet overview --wallet.name validator --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli wallet overview --wallet.name miner --subtensor.chain_endpoint ws://127.0.0.1:9946

# Run subnet miner and validator with pm2
echo "Running subnet miner and validator with pm2..."
pm2 start python --name miner -- neurons/miner.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name miner --wallet.hotkey default --logging.debug
pm2 start python --name validator -- neurons/validator.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name validator --wallet.hotkey default --logging.debug

# Register validator on root subnet and boost subnet
echo "Registering validator on root subnet and boosting subnet..."
btcli root register --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli root boost --netuid 1 --increase 1 --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946

echo "Setup complete! Verify your incentive mechanism in 72 minutes."