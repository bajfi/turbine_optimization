from setuptools import find_packages, setup

setup(
    name="turbine_optimization",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "pymoo>=0.6.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
        "xgboost>=1.5.0",
    ],
    python_requires=">=3.10",
)
