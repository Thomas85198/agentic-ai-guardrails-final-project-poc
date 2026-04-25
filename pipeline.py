"""
RAG Pipelines - baseline 與 guarded 兩個版本

兩版共用同一個 RAGCore（LlamaIndex + Qdrant + Ollama Qwen2.5）。
差異只有：
  - BaselineRAG：直接 query → 回答
  - GuardedRAG：INPUT 過 NeMo → query → OUTPUT 過 Presidio

這樣對照才公平，guardrails 是唯一的實驗變數。
"""
from dataclasses import dataclass, field
from typing import List, Optional

from rag_core import RAGCore, RetrievedChunk
from guardrails import (
    GuardrailClient,
    NemoInputGuard,
    PresidioOutputGuard,
)


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

    def __init__(self, rag: Optional[RAGCore] = None):
        self.rag = rag or RAGCore()

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
    ):
        self.rag = rag or RAGCore()
        self.input_guard = input_guard or NemoInputGuard()
        self.output_guard = output_guard or PresidioOutputGuard()

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
