from setuptools import setup, find_packages

setup(
    name="hybrid-llm-inference",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "pytest",
        "pytest-cov",
        "pyyaml",
        "tqdm",
        "psutil",
        "pynvml",
    ],
    python_requires=">=3.8",
)
