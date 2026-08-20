"""
TieredSummarizer — L0/L1/L2 分层摘要 (OpenViking 启发)
========================================================
OpenViking 思想: 每个条目写入时生成三层摘要, 读取时按任务深度加载:
  L0 abstract (~100 token): 一句话, 快速相关性检查
  L1 overview  (~2k token): 核心信息 + 使用场景, 规划用
  L2 details:            完整原始数据, 按需读取

AgentFrame 实现 (零额外 LLM 成本, 确定性启发式):
  - ingest 时同步生成 L0 (首句浓缩) + L1 (关键子句集合)
  - ask 时构建分层上下文: [L0 列表] → 命中块展开 L1 → 需要时 L2 全文
  - 与双轨压缩正交: 语义压缩(砍内容) × 物理压缩(砍体积) × 分层摘要(砍读取量)
"""
import re


def _split_sentences(text: str) -> list:
    """按中文/英文句号切句"""
    parts = re.split(r'[。！？!?；;]', text)
    return [p.strip() for p in parts if p.strip()]


class TieredSummarizer:
    """L0/L1/L2 分层摘要生成器 (确定性启发式, 无额外 LLM 调用)"""

    L0_MAX_CHARS = 40     # L0 一句话摘要上限
    L1_MAX_ITEMS = 5      # L1 关键子句条数

    @staticmethod
    def make_l0(text: str) -> str:
        """L0: 一句话摘要 — 首句浓缩 (快速相关性检查用)"""
        sentences = _split_sentences(text)
        if not sentences:
            return text[:TieredSummarizer.L0_MAX_CHARS]
        first = sentences[0]
        if len(first) <= TieredSummarizer.L0_MAX_CHARS:
            return first
        return first[:TieredSummarizer.L0_MAX_CHARS] + "…"

    @staticmethod
    def make_l1(text: str) -> list:
        """L1: 关键子句集合 — 覆盖主要信息点的概述"""
        sentences = _split_sentences(text)
        # 取前 N 条 + 含关键词的句子 (数字/对比/结论词)
        key_sents = sentences[:TieredSummarizer.L1_MAX_ITEMS]
        keywords = re.compile(r'(\d+|对比|结论|结果|实现|方法|方案|支持|不|最|×|x)')
        for s in sentences:
            if len(key_sents) >= TieredSummarizer.L1_MAX_ITEMS:
                break
            if s not in key_sents and keywords.search(s):
                key_sents.append(s)
        return key_sents[:TieredSummarizer.L1_MAX_ITEMS]

    @staticmethod
    def make_tiers(text: str) -> dict:
        """生成完整三层摘要结构"""
        return {
            "l0": TieredSummarizer.make_l0(text),
            "l1": TieredSummarizer.make_l1(text),
            "l2_len": len(text),  # L2 = 全文, 记录长度供决策
        }

    # ============ 分层上下文构建 ============

    @staticmethod
    def build_tiered_context(chunks: list, expand_l1: bool = True) -> str:
        """
        构建分层上下文 (OpenViking L0→L1→L2 按需展开):
        chunks: [(cid, meta_dict), ...] 按相关性排序
        输出: [L0 速览] → 命中块 L1 概述 → (L2 全文由调用方按需追加)
        """
        if not chunks:
            return "(无检索到相关知识块)"
        # L0 速览: 所有命中块的一句话摘要列表
        l0_lines = []
        for cid, meta in chunks:
            l0 = meta.get("l0") or TieredSummarizer.make_l0(meta.get("text", ""))
            l0_lines.append(f"  [{cid}] {l0}")
        l0_block = "【知识速览 (L0)】\n" + "\n".join(l0_lines)

        if not expand_l1:
            return l0_block

        # L1 概述: 每个命中块的关键子句 (比 L2 全文省 token)
        l1_lines = []
        for cid, meta in chunks:
            text = meta.get("text", "")
            l1 = meta.get("l1") or TieredSummarizer.make_l1(text)
            items = "；".join(l1) if l1 else text[:80]
            l1_lines.append(f"  [{cid}] {items}")
        l1_block = "\n【详情概述 (L1)】\n" + "\n".join(l1_lines)

        return l0_block + "\n" + l1_block
