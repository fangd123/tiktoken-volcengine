from setuptools import setup, find_namespace_packages

setup(
    name="volcengine_tiktoken_extension",
    version="0.1.0",
    description="Tiktoken extension for Volcengine Ark models tokenization",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_namespace_packages(include=['tiktoken_ext*']),
    install_requires=[
        "tiktoken",
        "volcenginesdkarkruntime",
        "requests"
    ],
    python_requires=">=3.6",
)
