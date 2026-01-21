import torch
import numpy as np
import argparse
import os
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

def generate_semantic_shuffle(model_name, n_channels, save_path, seed=42):
    """
    加载模型 Embedding，进行 K-means 聚类，并保存排序后的索引。
    """
    print(f"Loading model: {model_name}...")
    # 根据模型类型选择加载方式，通常现在的大模型都是 CausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    except:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    
    # 获取 Embedding 权重
    # 注意：不同模型的 embedding 层名称可能不同，这里涵盖了常见情况
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embeddings = model.model.embed_tokens.weight.detach().cpu().numpy()
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        embeddings = model.transformer.wte.weight.detach().cpu().numpy()
    else:
        raise ValueError("无法自动找到模型的 Embedding 层，请检查模型结构。")

    vocab_size = embeddings.shape[0]
    print(f"Vocab size: {vocab_size}, Embedding dim: {embeddings.shape[1]}")

    # 运行 K-Means
    print(f"Running K-Means (n_clusters={n_channels})... This may take a while.")
    kmeans = KMeans(n_clusters=n_channels, n_init=10, random_state=seed)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 构造排序索引
    # 我们希望生成的 shuffle 索引使得：token 在 shuffle 后的顺序是按聚类聚集的
    # 比如：[Cluster0的所有词, Cluster1的所有词, ...]
    
    # 创建 (label, original_index) 对
    token_info = []
    for idx, label in enumerate(cluster_labels):
        token_info.append((label, idx))
    
    # 排序：先按 Cluster ID 排序
    # 为了增加聚类内部的随机性（安全性），同 Cluster 内可以再次随机打乱，或者按原 ID 排序
    # 这里我们保持确定的排序，先按 Label，再按原 ID
    token_info.sort(key=lambda x: (x[0], x[1]))
    
    # 提取排序后的原 ID，这就是我们的 semantic_shuffle 映射
    sorted_indices = [x[1] for x in token_info]
    semantic_shuffle = torch.tensor(sorted_indices, dtype=torch.long)
    
    # 计算每个聚类的大小，用于后续动态切分
    cluster_counts = np.bincount(cluster_labels, minlength=n_channels)
    
    # 保存结果
    data = {
        "shuffle": semantic_shuffle,
        "counts": torch.tensor(cluster_counts, dtype=torch.long),
        "n_channels": n_channels,
        "model_name": model_name
    }
    torch.save(data, save_path)
    
    print("="*30)
    print(f"Done! Saved to {save_path}")
    print(f"Cluster sizes: {cluster_counts}")
    print(f"Std dev of sizes: {np.std(cluster_counts):.2f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate semantic clusters for MCMark.")
    parser.add_argument("--model_str", type=str, required=True, help="HuggingFace model path/name")
    parser.add_argument("--n_channels", type=int, default=20, help="Number of watermark channels (clusters)")
    parser.add_argument("--output", type=str, default="semantic_shuffle.pt", help="Path to save the .pt file")
    
    args = parser.parse_args()
    
    generate_semantic_shuffle(args.model_str, args.n_channels, args.output)