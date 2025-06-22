#!/usr/bin/env python3
"""
Setup script for LLM ARM Inference API
Handles platform-specific package installations
"""

import subprocess
import sys
import platform
import os

def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def get_platform_info():
    """Get platform information"""
    return {
        'system': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'is_arm': 'arm' in platform.machine().lower() or 'aarch64' in platform.machine().lower()
    }

def install_pytorch():
    """Install PyTorch with appropriate settings"""
    info = get_platform_info()
    print(f"Installing PyTorch for {info['machine']} architecture...")
    
    if info['is_arm']:
        # For ARM, use CPU-only PyTorch
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    else:
        # For x86_64, use default PyTorch
        cmd = "pip install torch torchvision torchaudio"
    
    return run_command(cmd)

def install_transformers():
    """Install transformers and related packages"""
    packages = [
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.90",
        "protobuf>=4.0.0,<5.0.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install '{package}'"):
            print(f"Failed to install {package}, trying without version constraint...")
            base_package = package.split('>=')[0].split('<')[0]
            run_command(f"pip install {base_package}", check=False)

def install_quantization():
    """Install quantization libraries if possible"""
    info = get_platform_info()
    
    if info['is_arm']:
        print("Installing bitsandbytes for ARM...")
        # Try ARM-compatible version
        success = run_command("pip install bitsandbytes>=0.40.0", check=False)
        if not success:
            print("Warning: Could not install bitsandbytes. Quantization will be disabled.")
    else:
        print("Installing bitsandbytes for x86_64...")
        run_command("pip install bitsandbytes>=0.40.0", check=False)

def install_web_framework():
    """Install FastAPI and related packages"""
    packages = [
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.20.0",
        "pydantic>=2.0.0",
        "aiofiles>=22.0.0",
        "python-multipart>=0.0.5"
    ]
    
    for package in packages:
        run_command(f"pip install '{package}'")

def install_utilities():
    """Install utility packages"""
    packages = [
        "psutil>=5.8.0",
        "numpy>=1.20.0"
    ]
    
    for package in packages:
        run_command(f"pip install '{package}'")

def install_llama_cpp():
    """Install llama-cpp-python with appropriate settings"""
    info = get_platform_info()
    
    if info['is_arm']:
        print("Installing llama-cpp-python with ARM NEON support...")
        cmd = "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
    else:
        print("Installing llama-cpp-python for x86_64...")
        cmd = "pip install llama-cpp-python"
    
    return run_command(cmd, check=False)

def main():
    """Main setup function"""
    print("=" * 50)
    print("LLM ARM Inference API Setup")
    print("=" * 50)
    
    info = get_platform_info()
    print(f"Platform: {info['system']} {info['machine']}")
    print(f"Python: {info['python_version']}")
    print(f"ARM architecture: {info['is_arm']}")
    print()
    
    # Upgrade pip first
    print("Upgrading pip...")
    run_command("pip install --upgrade pip setuptools wheel")
    
    # Install packages in order
    print("\n1. Installing PyTorch...")
    install_pytorch()
    
    print("\n2. Installing Transformers...")
    install_transformers()
    
    print("\n3. Installing web framework...")
    install_web_framework()
    
    print("\n4. Installing utilities...")
    install_utilities()
    
    print("\n5. Installing quantization libraries...")
    install_quantization()
    
    print("\n6. Installing llama.cpp (optional)...")
    install_llama_cpp()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    
    # Test imports
    print("\nTesting imports...")
    test_packages = [
        "torch",
        "transformers", 
        "fastapi",
        "uvicorn",
        "numpy",
        "psutil"
    ]
    
    for package in test_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Failed to import")
    
    print("\nYou can now run: python app/server.py")

if __name__ == "__main__":
    main()
