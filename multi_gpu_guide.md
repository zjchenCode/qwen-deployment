# Qwen2.5-VL-72B 多GPU配置指南

## 🚀 多GPU支持特性

### ✅ 已支持的功能
- ✅ 自动多GPU模型分布（`device_map="auto"`）
- ✅ 精确GPU内存控制（`MAX_MEMORY`环境变量）
- ✅ 多GPU内存监控和清理
- ✅ 负载均衡分配
- ✅ 跨GPU通信优化

### 📋 多GPU配置方式

#### 方式1: 自动分配（推荐）
```bash
export DEVICE_MAP="auto"
# 系统自动将模型层分布到可用GPU上
```

#### 方式2: 精确内存控制
```bash
# 为2×H100配置（每个GPU最大75GB）
export MAX_MEMORY="75GiB,75GiB"

# 为不均匀GPU配置
export MAX_MEMORY="70GiB,60GiB"
```

#### 方式3: 手动设备映射
```bash
export DEVICE_MAP='{"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 1, "lm_head": 1}'
```

## 🔧 H200 → 2×H100 迁移配置

### 推荐配置
```bash
# 基础设置
export MAX_MEMORY="75GiB,75GiB"  # 为每个H100预留5GB系统内存
export DEVICE_MAP="auto"
export LOAD_IN_4BIT="true"

# 优化设置
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export NCCL_P2P_DISABLE="1"  # 如果遇到P2P通信问题

# 内存管理（多GPU优化）
export AUTO_CLEANUP_MEMORY="true"
export CLEANUP_INTERVAL="2"      # 更频繁清理（多GPU需要）
export AGGRESSIVE_CLEANUP="true"
export FORCE_SYNC="true"
```

### 启动脚本
```bash
# 检查GPU状态
nvidia-smi

# 启动服务
cd /workspace
python3 server.py
```

## 📊 性能预期

### H200 vs 2×H100对比

| 指标 | H200 (144GB) | 2×H100 (160GB) |
|------|--------------|----------------|
| **可用内存** | ~135GB | ~150GB |
| **推理吞吐** | 1.0x | 0.9-1.1x |
| **复杂度** | 简单 | 中等 |
| **稳定性** | 高 | 高 |

### 预期改善
- ✅ **内存溢出问题解决**: 160GB vs 144GB
- ✅ **长文本处理能力提升**: 更大KV Cache空间
- ⚠️ **轻微延迟增加**: 跨GPU通信（5-10ms）
- ✅ **总体稳定性提升**: 内存压力降低

## 🔍 多GPU监控

### 内存监控API
```bash
# 查看各GPU使用情况
curl http://localhost:8000/health

# 详细多GPU指标
curl http://localhost:8000/metrics -H "Authorization: Bearer YOUR_API_KEY"
```

### 命令行监控
```bash
# 实时GPU监控
watch -n 1 nvidia-smi

# 详细内存分布
nvidia-sml --query-gpu=memory.used,memory.total --format=csv -l 1
```

## ⚠️ 注意事项

### 潜在问题
1. **GPU间通信延迟**: 首次推理可能较慢
2. **内存分布不均**: 需要调整MAX_MEMORY
3. **驱动兼容性**: 确保CUDA/驱动版本一致

### 解决方案
```bash
# 如果遇到NCCL错误
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# 如果遇到分布不均
export MAX_MEMORY="70GiB,80GiB"  # 手动调整

# 如果遇到同步问题
export CUDA_VISIBLE_DEVICES="0,1"
```

## 📈 最佳实践

### 1. 渐进式迁移
```bash
# 第一步：验证多GPU检测
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# 第二步：小模型测试
export MAX_MEMORY="40GiB,40GiB"

# 第三步：完整配置
export MAX_MEMORY="75GiB,75GiB"
```

### 2. 性能调优
```bash
# 优化CUDA内存分配
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:16"

# 优化多GPU通信
export NCCL_TREE_THRESHOLD=0
```

### 3. 故障排除
```bash
# 查看GPU拓扑
nvidia-smi topo -m

# 检查P2P连接
nvidia-smi nvlink --status

# 验证内存分配
python3 -c "
import torch
from transformers import AutoModel
print('GPU Memory before loading:')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.memory_allocated(i)/1024**3:.1f}GB')
"
```

## 🎯 推荐配置总结

对于您的情况，**强烈推荐使用2×H100**：

1. **解决内存不足**: 160GB > 144GB
2. **项目完全兼容**: 无需代码修改
3. **配置简单**: 只需设置环境变量
4. **性能提升**: 更大内存空间，更稳定运行

迁移成本极低，收益明显！ 