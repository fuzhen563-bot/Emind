"""
Emind 安装配置
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="emind",
    version="2.0.0",
    author="亦梓科技",
    description="亦梓·智脑 — 大语言模型训练推理框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["Emind", "Emind.*", "training", "training.*", "data_pipeline", "data_pipeline.*", "eval", "eval.*"]),
    entry_points={
        "console_scripts": [
            "emind=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "sentencepiece>=0.1.99",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.0",
        "jinja2>=3.1.0",
        "sse-starlette>=1.8.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "requests>=2.31.0",
        "pydantic>=2.10.0",
        "httpx>=0.25.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black>=23.0"],
        "vllm": ["vllm>=0.6.0"],
        "vllm-lora": ["vllm>=0.6.3"],
        "vllm-awq": ["vllm>=0.5.0", "autoawq"],
        "vllm-gptq": ["vllm>=0.5.0", "auto-gptq"],
        "vllm-full": ["vllm>=0.6.3", "autoawq", "auto-gptq"],
        "eval": ["scikit-learn>=1.3.0"],
    },
)
