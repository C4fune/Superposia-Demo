#!/usr/bin/env python3
"""
Setup configuration for the Next-Generation Quantum Computing Platform
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantum-platform",
    version="0.1.0",
    author="Quantum Platform Team",
    author_email="quantum@platform.dev",
    description="Next-Generation Quantum Computing Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantum-platform/quantum-platform",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Compilers",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "mypy>=0.991",
            "flake8>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantum-platform=quantum_platform.cli:main",
        ],
    },
) 