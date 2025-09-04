"""Setup script for mamba-kan package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [
        line.strip() 
        for line in f.readlines() 
        if line.strip() and not line.startswith("#")
    ]

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
with open(readme_path, encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mamba-kan",
    version="0.1.0",
    description="Factorial comparison of neural network architectures: MLP vs KAN × Transformer vs Mamba",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-username/Mamba_KAN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0", 
            "isort>=5.12.0",
            "flake8>=5.0.0",
        ],
        "profiling": [
            "fvcore",
            "memory-profiler",
            "py-spy",
        ]
    },
    entry_points={
        "console_scripts": [
            "mamba-kan-train=scripts.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)