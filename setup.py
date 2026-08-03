from setuptools import setup, find_packages

setup(
    name="rdt-wrapper",
    version="0.1.0",
    description="Recurrent-Depth Adaptation for Pretrained Transformers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jon Grayson",
    author_email="jon@graysondp.com",
    url="https://github.com/jgray4567/rdt-wrapper",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "numpy>=1.24",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-xdist"],
        "flash": ["flash-attn>=2.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)