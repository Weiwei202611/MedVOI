# MedVOI 代码包中文说明

这是论文 **Predicting patient-specific information value enables adaptive multimodal medical AI** 的参考实现。

核心流程：

1. 训练可接受任意非空模态子集的 task model；
2. 用患者级 5-fold cross-fitting 生成 out-of-fold 的 `Q_i(S)`；
3. 计算 `G_i(m;S)=Q_i(S∪{m})-Q_i(S)`；
4. 使用最终固定 task model 提取统一坐标系下的患者状态向量；
5. 训练 MedVOI，输入当前 state + candidate identity，输出 gain mean 和 gain uncertainty；
6. 用 heteroscedastic regression + pairwise ranking 训练；
7. 在验证集选择 STOP threshold 和风险惩罚；
8. 在锁定 test set 上做 one-step 与 sequential evaluation。

## 最重要的文件

- `configs/brats.yaml`：BraTS 主实验配置
- `scripts/train_task.py`：训练 task model
- `scripts/crossfit_gain_targets.py`：生成交叉拟合 gain supervision
- `scripts/extract_states.py`：生成 patient-state features
- `scripts/train_value.py`：训练 MedVOI
- `scripts/evaluate_one_step.py`：单步 gain/decision 评估
- `scripts/evaluate_sequential.py`：连续 reveal / STOP 评估
- `scripts/make_source_data.py`：导出图表底层数据

## 运行顺序

```bash
python scripts/validate_manifest.py --config configs/brats.yaml
python scripts/train_task.py --config configs/brats.yaml
python scripts/crossfit_gain_targets.py --config configs/brats.yaml
python scripts/extract_states.py --config configs/brats.yaml --checkpoint outputs/brats/task_final.pt
python scripts/train_value.py --config configs/brats.yaml
python scripts/evaluate_one_step.py --config configs/brats.yaml
python scripts/evaluate_sequential.py --config configs/brats.yaml
python scripts/make_source_data.py --config configs/brats.yaml
```

## 注意

论文没有写死的 optimizer、learning rate、batch size、hidden dimension 等参数，我已在 YAML 中给出可运行的参考默认值。若你已有真实实验配置，投稿/Zenodo 归档前应替换成真实执行值，不要把参考默认值当成已执行实验参数。
