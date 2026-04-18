import subprocess
import sys
import os
import importlib

def install_packages():
    print("--- 1. Installing Dependencies ---")
    requirements = [
        "torch==2.10.0",
        "transformers==4.57.6",
        "numpy==2.4.1",
        "tqdm==4.67.1",
        "tree-sitter==0.25.2",
        "tree-sitter-java==0.23.5",
        "tree-sitter-python==0.25.0",
        "tree-sitter-javascript==0.25.0",
        "tree-sitter-cpp==0.23.4"
        # Note: tree-sitter-erlang is not available on PyPI, so you can install it manually from the GitHub repository:
        # https://github.com/the-mikedavis/tree-sitter-erlang/tarball/master
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + requirements)
        print("\n✓ All packages installed successfully.")
    except Exception as e:
        print(f"\n[!] Installation failed: {e}")

def check_runtime():
    print("\n--- 2. Checking Runtime Environment ---")
    
    # Check PyTorch & GPU
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA Available: Yes ({torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) Available: Yes")
        else:
            print("! GPU Acceleration: No (Running on CPU)")
    except ImportError:
        print("× PyTorch not found.")

    # Check Transformers
    try:
        from transformers import RobertaTokenizer
        RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        print("✓ HuggingFace Transformers & Tokenizer: Ready")
    except Exception as e:
        print(f"× Transformers error: {e}")

    # Check Tree-sitter Parsers
    print("\n--- 3. Checking Tree-sitter Parsers ---")
    langs = {
        'java': 'tree_sitter_java',
        'python': 'tree_sitter_python',
        'javascript': 'tree_sitter_javascript',
        'cpp': 'tree_sitter_cpp',
    }
    
    for name, lib in langs.items():
        try:
            mod = importlib.import_module(lib)
            print(f"✓ {name.upper():<10}: Installed")
        except ImportError:
            print(f"× {name.upper():<10}: Missing")

def verify_project_structure():
    print("\n--- 4. Verifying Project Structure ---")
    required_files = ['model.py', 'evaluate.py', 'train.py', 'config.json']
    for f in required_files:
        if os.path.exists(f):
            print(f"✓ Found {f}")
        else:
            print(f"× Missing {f} (Ensure you are in the project root)")

if __name__ == "__main__":
    install_packages()
    check_runtime()
    verify_project_structure()
    print("\nSetup complete. If all checks passed, you are ready to train.")