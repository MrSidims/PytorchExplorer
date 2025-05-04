#!/bin/bash
set -e

echo "Updating APT and installing Node.js (v20)..."
sudo apt update
sudo apt install -y curl ca-certificates gnupg

curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

echo "Installing frontend dependencies..."
npm install

echo "Frontend setup complete."

