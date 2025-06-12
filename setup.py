from setuptools import setup, find_packages

setup(
    name="shopper-insights",
    version="1.0.0",
    description="Customer Segmentation using K-Means Clustering and Machine Learning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.17.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)