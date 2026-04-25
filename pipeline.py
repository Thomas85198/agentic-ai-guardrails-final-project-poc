"""
RAG Pipelines - baseline 與 guarded 兩個版本

兩版共用同一個 RAGCore（LlamaIndex + Qdrant + Ollama Qwen2.5）。
差異只有：
  - BaselineRAG：直接 query → 回答
  - GuardedRAG：INPUT 過 NeMo → query → OUTPUT 過 Presidio

這樣對照才公平，guardrails 是唯一的實驗變數。

Stage A：summary 類 query 走 Paper Card 快速路徑（離線預先 build 好的論文卡），
        不再硬走 RAG 切片，徹底解決切片無法 summary 的問題。
Stage B：Router agent 把 user query 分到 4 個 intent (summary / qa / recommend / out_of_scope),
        取代原本的 keyword hack。Recommender agent 處理跨論文推薦。
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
from agents import Recommender, Router, RouterDecision


# === Paper Card 整合 ===
DEFAULT_CARDS_DIR = "data/cards"

# Router 判 out_of_scope 時的回覆
OUT_OF_SCOPE_REPLY = (
    "這個問題超出本系統的論文檢索範圍。"
    "本系統專注於協助你查詢、摘要、比較目前已索引的論文。"
    "你可以試試：詢問某篇論文的概要、實驗細節，或描述你的研究方向尋求推薦。"
)


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


def _resolve_summary_card(
    decision: RouterDecision, cards: Dict[str, dict]
) -> Optional[dict]:
    """summary intent 應該指名一篇。Router 沒指名時:單篇就用唯一一篇,多篇放棄。"""
    if decision.paper_ids:
        return cards.get(decision.paper_ids[0])
    if len(cards) == 1:
        return next(iter(cards.values()))
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
    """純 RAG，無任何 guardrails（Stage B：仍走 Router agent 分流）"""

    def __init__(
        self,
        rag: Optional[RAGCore] = None,
        cards: Optional[Dict[str, dict]] = None,
        router: Optional[Router] = None,
        recommender: Optional[Recommender] = None,
    ):
        self.rag = rag or RAGCore()
        self.cards = cards if cards is not None else load_cards()
        self.router = router or Router()
        self.recommender = recommender or Recommender()

    def __call__(self, query: str) -> PipelineResult:
        answer, chunks = self.rag.query(query)
        return PipelineResult(
            query=query,
            final_response=answer,
            retrieved_chunks=chunks,
        )

    def chat(self, messages: list) -> PipelineResult:
        """多輪對話版本（in-context learning）。messages 由前端維護。

        Stage B 流程：
          Router → {summary | qa | recommend | out_of_scope} → 對應 handler
        """
        latest = messages[-1]["content"] if messages else ""
        decision = self.router.decide(latest, self.cards)
        print(f"[BaselineRAG] router → {decision.intent} {decision.paper_ids} ({decision.reasoning})")

        if decision.intent == "out_of_scope":
            return PipelineResult(
                query=latest,
                final_response=OUT_OF_SCOPE_REPLY,
                retrieved_chunks=[],
            )

        if decision.intent == "summary":
            card = _resolve_summary_card(decision, self.cards)
            if card is not None:
                return PipelineResult(
                    query=latest,
                    final_response=card_to_markdown(card),
                    retrieved_chunks=[],
                )
            # 卡片找不到就 fallback 到 qa

        if decision.intent == "recommend":
            scope = (
                {pid: self.cards[pid] for pid in decision.paper_ids if pid in self.cards}
                or self.cards
            )
            reply = self.recommender.recommend(latest, scope)
            return PipelineResult(
                query=latest,
                final_response=reply,
                retrieved_chunks=[],
            )

        # qa（含 summary fallback）
        answer, chunks = self.rag.chat(messages, paper_ids=decision.paper_ids or None)
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
        router: Optional[Router] = None,
        recommender: Optional[Recommender] = None,
    ):
        self.rag = rag or RAGCore()
        self.input_guard = input_guard or NemoInputGuard()
        self.output_guard = output_guard or PresidioOutputGuard()
        self.cards = cards if cards is not None else load_cards()
        self.router = router or Router()
        self.recommender = recommender or Recommender()

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
        """多輪對話版本：NeMo + Router agent + Presidio 三層串接。

        Stage B 流程：
          1. NeMo INPUT       → 擋 prompt injection / 離題
          2. Router agent     → 分到 summary / qa / recommend / out_of_scope
          3. 各 intent 的 handler 跑（qa 還會過 Presidio INPUT 遮罩 + OUTPUT 兜底）
          4. Presidio OUTPUT  → 所有 handler 的回應都過一遍兜底

        注意：NeMo 只看 messages[-1]，無法偵測 multi-turn jailbreak
        （見 report.md 第 8 節限制 8）。
        """
        latest = messages[-1]["content"] if messages else ""

        # === Step 1: NeMo INPUT ===
        nemo_check = self.input_guard.apply_guardrail("INPUT", latest)
        if nemo_check["action"] == "GUARDRAIL_INTERVENED":
            return PipelineResult(
                query=latest,
                final_response=nemo_check["output"][0]["text"],
                blocked_at="INPUT",
                block_reason=str(nemo_check["assessments"]),
                retrieved_chunks=[],
            )

        # === Step 2: Router agent ===
        decision = self.router.decide(latest, self.cards)
        print(f"[GuardedRAG] router → {decision.intent} {decision.paper_ids} ({decision.reasoning})")

        # === Step 3: 各 intent 處理 ===
        chunks: List[RetrievedChunk] = []

        if decision.intent == "out_of_scope":
            return PipelineResult(
                query=latest,
                final_response=OUT_OF_SCOPE_REPLY,
                retrieved_chunks=[],
            )

        elif decision.intent == "summary":
            card = _resolve_summary_card(decision, self.cards)
            if card is not None:
                raw_answer = card_to_markdown(card)
            else:
                # 找不到對應卡片 → fallback 到 qa
                raw_answer, chunks = self._guarded_qa(messages, decision.paper_ids)

        elif decision.intent == "recommend":
            scope = (
                {pid: self.cards[pid] for pid in decision.paper_ids if pid in self.cards}
                or self.cards
            )
            raw_answer = self.recommender.recommend(latest, scope)

        else:  # qa
            raw_answer, chunks = self._guarded_qa(messages, decision.paper_ids)

        # === Step 4: Presidio OUTPUT 兜底（所有路徑都過）===
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

    def _guarded_qa(
        self, messages: list, paper_ids: List[str]
    ) -> tuple[str, List[RetrievedChunk]]:
        """qa 路徑：Presidio INPUT 遮罩 → RAG (filter by paper_ids) → 回答。
        OUTPUT guard 由 chat() 統一處理。
        """
        latest = messages[-1]["content"] if messages else ""
        pii_in = self.output_guard.apply_guardrail("INPUT", latest)
        if pii_in["action"] == "GUARDRAIL_INTERVENED":
            redacted_latest = pii_in["output"][0]["text"]
            messages_for_llm = messages[:-1] + [
                {"role": "user", "content": redacted_latest}
            ]
        else:
            messages_for_llm = messages

        return self.rag.chat(messages_for_llm, paper_ids=paper_ids or None)


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
