# exp-sign-probe:D2/D3 影响力符号探针(Amarel 版)

> 对应 `exp-0715/implementation_diagnosis_plan.md` 的 **D2、D3** 两个实验,零训练、纯推理。
> 回答一个问题:**训练时用 G=8 小陪审团估出的逐 token 影响力符号,到底准不准?**
> 背景与判读依据见 `exp-0715/discussion_notes_20260715.md` §1–§2 与复盘仪表盘。

## 实验内容

- **参考真值**:每道探针题采 **R=256** 条 rollout(温度 1.0,生成上限已放宽到 **1024**,训练时是 512/768),用全体 256 条做大陪审团 + 精确留一,得到每个 token 的参考影响力符号;并用 128 vs 128 两半检验参考自身是否稳定。
- **D2(符号准确率 vs 陪审团大小)**:从 256 条里重采 G ∈ {8, 16, 32, 64} 的小陪审团(每档 32 个),**完全按训练的算法**(组内均值优势、精确留一、奇偶分半门控)估符号,报告与参考真值的**质量加权(|d·ι|)一致率**、十分位校准、按 token 类型分解、退化陪审团率、以及门控保留子集的准确率。
- **D3(分半一致率的水分)**:模拟 200 个训练组(G=8),对每个 token 算三票——
  票1 全组留一(训练的 `iota`)、票2 对面半批(训练的 `iota_alt`)、票3 本半批去己(与票2 真正不相交)。
  报告 agree12(应复现训练日志的 ~0.76,校准点)、agree32(真独立一致率)、水分 = agree12−agree32,以及各票对参考真值的真实准确率。

**预写的判读标准**(自动写进 `results/report.md`):G=8 质量加权准确率 <60% ⇒ F1 实锤,先修估计器;60–70% ⇒ 看大 G 是否抬升;≥70% ⇒ 符号可用,问题在别处(F2/F3/F6)。

- **块级重分析(B0–B2,2026-07-16 新增)**:纯离线、只读已有 `tokens.npz`/`d2.npz`,验证"分块监督"提案(块内共享符号,信噪比 ~√块长)。B0 真符号成片性(相邻一致率、游程、coherence(T));B2 块级准确率/实现增益率 vs (G × 块方案,含换行分段与整条 rollout);B2 置信度–覆盖率曲线(低置信块回退 0 / 回退标准 OPD),并报告 OPD 基线对齐率作对照。登录节点 CPU 几分钟:

  ```bash
  python -m probe.block_analysis --config config.yaml --out /scratch/$USER/opd_sign_probe/run1
  # 结果:<out>/results/block_report.md + block_summary.json(中文自动判读)
  ```

## 目录

```
config.yaml                  常改参数(数据集、R、陪审团档位、生成上限、adapter)
probe/
  config.py                  全部配置 + manifest 落盘(防 F9 配置漂移)
  verifier.py                判分器(与训练脚本逐行一致)
  data.py                    数据加载 + 题目粗筛(选通过率接近 0.5 的混合题)
  rollout.py                 阶段1:vLLM 采样(粗筛 64 题×16 → 选 8 题×256)
  influence.py               核心数学:G 累加 / ι 打分 / 精确留一(纯函数)
  stages.py                  阶段2–4:teacher d_t → 参考真值 → D2 → D3
  analysis.py                阶段5:汇总 → summary.json + report.md(中文自动判读)
  run_probe.py               编排入口(各阶段有产物即跳过,可断点续跑)
tests/test_influence_small.py  暴力对照单元测试(5 项恒等式,CPU 秒级)
slurm/                       Amarel 脚本:下载、环境、冒烟、正式
```

## Amarel 上手(5 步)

```bash
# 0) 把本目录传上去
scp -r exp-sign-probe amarel:/scratch/$USER/

# 1) 登录节点:装环境(一次)
cd /scratch/$USER/exp-sign-probe && bash slurm/setup_env.sh

# 2) 登录节点:下载模型+数据到 scratch(计算节点无外网)
conda activate opd-probe && bash slurm/00_download.sh

# 3) 冒烟测试(2 题×32 条,约 10–20 分钟)——通过后再跑正式
sbatch slurm/submit_smoke.sbatch

# 4) 正式实验(8 题×256 条 + 全部陪审团扫描)
sbatch slurm/submit_probe.sbatch

# 5) 看结果
cat /scratch/$USER/opd_sign_probe/run1/results/report.md
```

**资源**:单卡(建议 ≥24GB 显存;A100 最佳,V100 会自动落到 fp16)。
**时长估计**:采样 ~0.5h;teacher+参考 ~1h;D2 陪审团扫描 2–4h;D3 ~0.5h;合计 4–6h(sbatch 给了 12h 余量)。分区名 `gpu` 请按课题组配额自行修改。

## 输出

- `manifest.json` / `versions.json`:解析后的完整配置与包版本(可复现性);
- `screen.json` / `questions.json`:粗筛通过率与选中的探针题;
- `q*/rollouts.jsonl`、`meta.json`:每题 256 条采样(token id、判分、是否截断);
- `q*/tokens.npz`:每个 token 的参考 ι、教师信号 d、类型、参考两半符号;
- `q*/d2.npz` + `d2_juries.json`、`q*/d3.npz`:陪审团级原始结果;
- `results/summary.json` + **`results/report.md`**:最终报告(含预写判读)。

## 与训练日志的校准点

D3 的 **agree12** 就是训练日志里的 `agree` 列(基座模型上,DeepMath 战役第 10 步记录为 0.77)。
若本探针在基座模型上给出 agree12 ≈ 0.76–0.78,说明对训练机制的复现是忠实的——
此时 acc1(真准确率)与 agree12 的差距,就是之前被"两票互相印证"掩盖的部分。

## 探训练中期检查点(可选,第二步)

把 Drive 上的某个 `step{N}/adapter/` 目录拷到 Amarel,然后:
```bash
python -m probe.run_probe --config config.yaml --out /scratch/$USER/opd_sign_probe/step300 \
    --adapter /path/to/step300/adapter --stage all
```
(vLLM 与 HF 前向都会挂上该 LoRA。)基座 + 中期各跑一次,可看符号质量随训练的变化。

## 注意事项

- **温度必须 1.0**(采样分布 = 被分析的策略,见讨论纪要 §3),config 里请勿改;
- 生成上限默认 1024(**已按讨论放宽**;`meta.json` 里的 `trunc_rate` 应 <5%,不然再调到 2048);
- `include_truncated: true` 为训练忠实口径;想看"干净陪审团"可改 false 重跑 reference 之后的阶段;
- 已在本地验证:`tests/` 5 项恒等式(含精确留一)全过;analysis 管道用合成数据端到端校验,
  植入的真值(准确率/退化率/一致率)均被正确恢复;
- 计算节点离线:sbatch 里已设 `HF_HUB_OFFLINE=1` 等,前提是第 2 步下载完成。
