#!/usr/bin/env python3
"""
Emind 数据管线使用示例
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import DataCollector, DataCleaner, DataSynthesizer, DataFormatter, DatasetManager


def demonstrate_pipeline():
    print("=" * 50)
    print("Emind 数据管线用法示例")
    print("=" * 50)

    # 1. Dataset Manager — 版本管理
    print("\n[1] DatasetManager — 数据集注册与版本管理")
    manager = DatasetManager(base_dir="data")
    print(f"    Raw files: {len(manager.list_raw())}")
    print(f"    Processed files: {len(manager.list_processed())}")
    print(f"    Versions: {len(manager.list_versions())}")

    # 2. Data Collector — 多源采集
    print("\n[2] DataCollector — 多源数据采集")
    collector = DataCollector(data_dir="data")
    sources = collector.list_sources()
    print(f"    Available sources: {len(sources)}")
    for s in sources[:5]:
        print(f"      - {s}")

    # 3. Data Cleaner — 数据清洗
    print("\n[3] DataCleaner — 数据清洗")
    cleaner = DataCleaner(min_length=10, max_length=8192)
    sample_texts = [
        "你好，请问有什么可以帮助你的？" * 50,
        "哈哈哈aaaaaaaaaaaaaaaaaaaa",
        "测试数据" * 1000,
        "这是一个正常的中文句子。",
    ]
    for t in sample_texts:
        score = cleaner._quality_score(t)
        lang = cleaner.detect_language(t)
        pii_stripped = cleaner.strip_pii("联系方式：13800138000，邮箱：test@example.com")
        print(f"    质量: {score:.2f}, 语言: {lang}")
    print(f"    PII脱敏示例: {pii_stripped[:50]}...")

    # 4. Data Synthesizer — 数据合成
    print("\n[4] DataSynthesizer — 数据合成")
    synthesizer = DataSynthesizer(seed=42)
    synthetic = synthesizer.generate_from_template("qa", count=5)
    for item in synthetic[:3]:
        print(f"    Q: {item['prompt'][:50]}...")
        print(f"    A: {item['response'][:50]}...")
        print()

    # 5. Data Formatter — 格式化输出
    print("\n[5] DataFormatter — 格式化")
    formatter = DataFormatter(system_prompt="你是一个有用的助手。")
    sft_data = formatter.to_sft(synthetic[:3])
    print(f"    SFT格式: {len(sft_data)} samples")
    print(f"    Sample messages: {len(sft_data[0]['messages'])} turns")

    # 6. 完整管线 CLI
    print("\n[6] 命令行用法")
    print("    python cli.py pipeline --collect data/raw --process --format sft")
    print("    python cli.py pipeline --collect data/raw --process --format dpo --lang zh")

    print("\n" + "=" * 50)
    print("数据管线示例完毕")
    print("=" * 50)


if __name__ == "__main__":
    demonstrate_pipeline()
