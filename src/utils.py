"""
Utility functions for SSL fix, random seed setting, logging, and environment configuration.
Must apply SSL fix before importing HuggingFace datasets/aiohttp.
"""
import ssl

def apply_ssl_fix():
    """Apply Windows SSL certificate patch for HuggingFace datasets/aiohttp."""
    try:
        ssl.SSLContext.load_default_certs = lambda self, purpose=None: None
    except Exception:
        pass

# Automatically execute SSL fix upon module import
apply_ssl_fix()

import os
import random
import sys
import logging
import numpy as np
import torch

def setup_environment():
    """Ensure environment is configured cleanly."""
    apply_ssl_fix()

def set_seed(seed: int = 42):
    """Set random seed across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Return PyTorch device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logger(name: str = "pharma_assistant") -> logging.Logger:
    """Configure and return a structured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
