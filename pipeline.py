"""
RAG Pipelines - baseline 與 guarded 兩個版本

兩版共用同一個 RAGCore（LlamaIndex + Qdrant + Ollama Qwen2.5）。
差異只有：
  - BaselineRAG：直接 query → 回答
  - GuardedRAG：INPUT 過 NeMo → query → OUTPUT 過 Presidio

這樣對照才公平，guardrails 是唯一的實驗變數。

Stage A 額外能力：summary 類 query 走 Paper Card 快速路徑（離線預先 build 好的論文卡），
不再硬走 RAG 切片，徹底解決切片無法 summary 的問題。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from rag_core import RAGCore, RetrievedChunk
from guardrails import (
    GuardrailClient,
    NemoInputGuard,
    PresidioOutputGuard,
)


# === Paper Card 整合 ===
DEFAULT_CARDS_DIR = "data/cards"

# Stage A：簡單 keyword 偵測。Stage B 會換成 LLM Router。
SUMMARY_KEYWORDS = (
    "摘要", "簡述", "簡介", "概要", "概述", "總結",
    "主要貢獻", "主要結果", "主要發現", "主要研究",
    "介紹這篇", "這篇論文是什麼", "這篇論文在做什麼",
    "summary", "summarise", "summarize", "overview",
)


def is_summary_query(text: str) -> bool:
    lower = text.lower()
    return any(kw.lower() in lower for kw in SUMMARY_KEYWORDS)


def load_cards(cards_dir: str = DEFAULT_CARDS_DIR) -> Dict[str, dict]:
    """讀 data/cards/*.json → {doc_id: card}。資料夾不存在或檔案壞掉時靜默略過。"""
    p = Path(cards_dir)
    if not p.is_dir():
        return {}
    out: Dict[str, dict] = {}
    for f in sorted(p.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[pipeline] 跳過壞掉的卡片 {f.name}：{e}")
    return out


def card_to_markdown(card: dict) -> str:
    """把 card dict 轉成易讀 markdown，給使用者看。"""
    parts: List[str] = []
    title = card.get("title")
    if title:
        parts.append(f"## {title}")
    if card.get("problem"):
        parts.append(f"**研究問題**：{card['problem']}")
    if card.get("method"):
        parts.append(f"**方法**：{card['method']}")
    ds = card.get("datasets")
    if ds:
        ds_str = ", ".join(ds) if isinstance(ds, list) else str(ds)
        parts.append(f"**資料集**：{ds_str}")
    if card.get("key_findings"):
        parts.append(f"**主要結果**：{card['key_findings']}")
    contribs = card.get("contributions")
    if contribs:
        items = contribs if isinstance(contribs, list) else [contribs]
        parts.append("**研究貢獻**：\n" + "\n".join(f"- {c}" for c in items))
    lims = card.get("limitations")
    if lims:
        items = lims if isinstance(lims, list) else [lims]
        parts.append("**限制**：\n" + "\n".join(f"- {x}" for x in items))
    if card.get("applicable_for"):
        parts.append(f"**適用情境**：{card['applicable_for']}")
    return "\n\n".join(parts) if parts else "（卡片內容為空）"


def pick_card_for_query(query: str, cards: Dict[str, dict]) -> Optional[dict]:
    """目前單篇場景：唯一一張就直接用；多篇場景留給 Stage B 的 Router 決定。

    多篇先支援一招：若 query 中明確點名某 doc_id（檔名 stem），就用該卡。
    """
    if not cards:
        return None
    if len(cards) == 1:
        return next(iter(cards.values()))
    lower = query.lower()
    for doc_id, card in cards.items():
        if doc_id.lower() in lower:
            return card
    return None


@dataclass
class PipelineResult:
    """單次查詢的結果，含中間步驟資訊用於分析"""
    query: str
    final_response: str
    blocked_at: Optional[str] = None  # "INPUT" / "OUTPUT" / None
    block_reason: Optional[str] = None
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)

    @property
    def retrieved_chunk_ids(self) -> List[str]:
        """向後相容：原本對外暴露的是 chunk_id list，給舊 evaluator/report 用。"""
        return [c.chunk_id for c in self.retrieved_chunks]


class BaselineRAG:
    """純 RAG，無任何 guardrails"""

    def __init__(
        self,
        rag: Optional[RAGCore] = None,
        cards: Optional[Dict[str, dict]] = None,
    ):
        self.rag = rag or RAGCore()
        self.cards = cards if cards is not None else load_cards()

    def __call__(self, query: str) -> PipelineResult:
        answer, chunks = self.rag.query(query)
        return PipelineResult(
            query=query,
            final_response=answer,
            retrieved_chunks=chunks,
        )

    def chat(self, messages: list) -> PipelineResult:
        """多輪對話版本（in-context learning）。messages 由前端維護。"""
        latest = messages[-1]["content"] if messages else ""

        # Stage A: summary 類 query → 直接回 Paper Card
        if is_summary_query(latest):
            card = pick_card_for_query(latest, self.cards)
            if card is not None:
                return PipelineResult(
                    query=latest,
                    final_response=card_to_markdown(card),
                    retrieved_chunks=[],
                )

        answer, chunks = self.rag.chat(messages)
        return PipelineResult(
            query=latest,
            final_response=answer,
            retrieved_chunks=chunks,
        )


class GuardedRAG:
    """加上 input/output 雙閘 guardrails 的 RAG"""

    def __init__(
        self,
        rag: Optional[RAGCore] = None,
        input_guard: Optional[GuardrailClient] = None,
        output_guard: Optional[GuardrailClient] = None,
        cards: Optional[Dict[str, dict]] = None,
    ):
        self.rag = rag or RAGCore()
        self.input_guard = input_guard or NemoInputGuard()
        self.output_guard = output_guard or PresidioOutputGuard()
        self.cards = cards if cards is not None else load_cards()

    def __call__(self, query: str) -> PipelineResult:
        # === Step 1: NeMo 檢查 INPUT（prompt injection / 離題）===
        in_check = self.input_guard.apply_guardrail("INPUT", query)
        if in_check["action"] == "GUARDRAIL_INTERVENED":
            return PipelineResult(
                query=query,
                final_response=in_check["output"][0]["text"],
                blocked_at="INPUT",
                block_reason=str(in_check["assessments"]),
                retrieved_chunks=[],
            )

        # === Step 2: Presidio 檢查 INPUT（PII 遮罩）===
        # v2 雙向防禦：把 PII 在送進 LLM/retrieval 之前就遮掉，
        # 確保 LLM context、Ollama log、embedding 都看不到原始 PII。
        pii_in = self.output_guard.apply_guardrail("INPUT", query)
        effective_query = (
            pii_in["output"][0]["text"]
            if pii_in["action"] == "GUARDRAIL_INTERVENED"
            else query
        )

        # === Step 3: 用遮罩後的 query 跑 RAG ===
        raw_answer, chunks = self.rag.query(effective_query)

        # === Step 4: Presidio 檢查 OUTPUT（兜底，防 LLM 從 chunks 撈出 PII）===
        out_check = self.output_guard.apply_guardrail("OUTPUT", raw_answer)
        if out_check["action"] == "GUARDRAIL_INTERVENED":
            return PipelineResult(
                query=query,
                final_response=out_check["output"][0]["text"],
                blocked_at="OUTPUT",
                block_reason=str(out_check["assessments"]),
                retrieved_chunks=chunks,
            )

        return PipelineResult(
            query=query,
            final_response=raw_answer,
            retrieved_chunks=chunks,
        )

    def chat(self, messages: list) -> PipelineResult:
        """多輪對話版本：INPUT guard 看最新 query，OUTPUT guard 看 LLM 回應。

        v2 雙向 PII 防禦：
          1. NeMo INPUT  → 擋 prompt injection / 離題
          2. Presidio INPUT → 遮罩 PII（**遮罩後的 query 才送 RAG**）
          3. RAG（看不到原始 PII）
          4. Presidio OUTPUT → 兜底，防 LLM 從 chunks 撈出 PII

        注意：NeMo 只看 messages[-1]，無法偵測 multi-turn jailbreak
        （見 report.md 第 8 節限制 8）。
        """
        latest = messages[-1]["content"] if messages else ""

        # === Step 1: NeMo INPUT（prompt injection / 離題）===
        nemo_check = self.input_guard.apply_guardrail("INPUT", latest)
        if nemo_check["action"] == "GUARDRAIL_INTERVENED":
            return PipelineResult(
                query=latest,
                final_response=nemo_check["output"][0]["text"],
                blocked_at="INPUT",
                block_reason=str(nemo_check["assessments"]),
                retrieved_chunks=[],
            )

        # === Stage A: summary 類 query → 直接回 Paper Card（仍過 OUTPUT guard 兜底）===
        if is_summary_query(latest):
            card = pick_card_for_query(latest, self.cards)
            if card is not None:
                card_text = card_to_markdown(card)
                out_check = self.output_guard.apply_guardrail("OUTPUT", card_text)
                final = (
                    out_check["output"][0]["text"]
                    if out_check["action"] == "GUARDRAIL_INTERVENED"
                    else card_text
                )
                return PipelineResult(
                    query=latest,
                    final_response=final,
                    retrieved_chunks=[],
                )

        # === Step 2: Presidio INPUT PII 遮罩（不擋下，遮罩後續送）===
        pii_in = self.output_guard.apply_guardrail("INPUT", latest)
        if pii_in["action"] == "GUARDRAIL_INTERVENED":
            redacted_latest = pii_in["output"][0]["text"]
            # 用遮罩版的最新 user message 重組 messages，前面歷史保留原樣
            messages_for_llm = messages[:-1] + [
                {"role": "user", "content": redacted_latest}
            ]
        else:
            messages_for_llm = messages

        # === Step 3: RAG（用遮罩版送進 LLM 與 retrieval）===
        raw_answer, chunks = self.rag.chat(messages_for_llm)

        # === Step 4: Presidio OUTPUT（兜底）===
        out_check = self.output_guard.apply_guardrail("OUTPUT", raw_answer)
        if out_check["action"] == "GUARDRAIL_INTERVENED":
            return PipelineResult(
                query=latest,
                final_response=out_check["output"][0]["text"],
                blocked_at="OUTPUT",
                block_reason=str(out_check["assessments"]),
                retrieved_chunks=chunks,
            )

        return PipelineResult(
            query=latest,
            final_response=raw_answer,
            retrieved_chunks=chunks,
        )


if __name__ == "__main__":
    print("初始化 RAGCore...")
    shared_rag = RAGCore()  # 共用一份，省一半建索引時間

    print("初始化 BaselineRAG...")
    baseline = BaselineRAG(rag=shared_rag)
    print("初始化 GuardedRAG（會載入 NeMo + Presidio）...")
    guarded = GuardedRAG(rag=shared_rag)

    test = "本論文的實驗結果如何？"
    print(f"\n查詢：{test}")
    print(f"\n[Baseline] {baseline(test).final_response[:300]}")
    print(f"\n[Guarded]  {guarded(test).final_response[:300]}")
