from setuptools import setup, find_packages

setup(
    name="hybrid_llm_inference",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "transformers",
        "pytest",
    ],
)
