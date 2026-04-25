"""
Paper Card Builder - 離線為每篇 PDF 產出一張結構化「論文卡」

設計：
  - 短論文（< STUFF_LIMIT 字）：一次塞進 Qwen2.5:14b 出卡
  - 長論文：map-reduce
      Map 階段：依頁切成多個 chunk，每個 chunk 抽出零碎事實（partial JSON）
      Reduce 階段：把所有 partial 合併成最終 card
  - 跳過目錄頁（沿用 rag_core.load_pdf_pages）
  - 輸出落到 data/cards/<doc_id>.json，方便人工審視與後續 Router/Synthesizer 取用

執行：
  python build_cards.py            # 為 data/ 下所有 PDF 產卡（已存在則跳過）
  python build_cards.py --force    # 強制重建所有卡
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

from llama_index.llms.ollama import Ollama

from rag_core import DEFAULT_DATA_DIR, DEFAULT_LLM_MODEL, load_pdf_pages


CARD_DIR_NAME = "cards"

# 一次塞入的字數上限（短於此就走 stuff 路徑，省一次 reduce call）
STUFF_LIMIT_CHARS = 18000

# Map 階段每個 chunk 的字數上限（給 prompt 與輸出留 buffer）
MAP_CHUNK_CHARS = 14000


# === Prompts ===

STUFF_PROMPT = """你是一個學術論文分析助手。請閱讀以下論文全文，輸出一張結構化的「論文卡」。

**輸出格式**：純 JSON，不要任何 markdown code fence、不要前後說明文字。
**語言**：所有欄位用繁體中文，簡潔不囉唆。
**規則**：欄位內容必須來自論文本身，不要外推、不要編造；若論文沒提到某欄位就填 "未提及"。

JSON schema：
{{
  "title": "論文標題",
  "problem": "這篇論文要解決什麼問題（1-2 句）",
  "method": "採用的方法或技術概要（2-3 句，可列出關鍵模組）",
  "datasets": ["使用的資料集名稱"],
  "key_findings": "主要實驗結果或發現（2-3 句，含關鍵指標）",
  "contributions": ["貢獻 1", "貢獻 2", "貢獻 3"],
  "limitations": ["限制 1", "限制 2"],
  "applicable_for": "這篇論文適合什麼類型的研究參考（1-2 句，給後續論文推薦用）"
}}

=== 論文全文 ===
{paper_text}
=== END 論文全文 ===

請現在輸出 JSON："""


MAP_PROMPT = """你正在從論文的一個段落中抽取事實線索。**只根據以下這個段落**，沒提到的欄位給空陣列 []。

**輸出格式**：純 JSON，不要 markdown code fence、不要前後說明文字。
**語言**：繁體中文。
**規則**：每條線索一句話，附上原文出處頁碼如 (p.5)；不要外推。

段落範圍：p.{first_page}–p.{last_page}

JSON schema：
{{
  "title": "若段落中明確出現論文標題就寫，否則空字串",
  "problem_clues": ["問題定義線索 (p.X)"],
  "method_clues": ["方法/模組/架構描述 (p.X)"],
  "dataset_clues": ["資料集名稱 (p.X)"],
  "finding_clues": ["實驗結果或數字 (p.X)"],
  "contribution_clues": ["明確聲稱的貢獻 (p.X)"],
  "limitation_clues": ["明確聲稱的限制或未來工作 (p.X)"]
}}

=== 段落內容 ===
{chunk_text}
=== END 段落 ===

請現在輸出 JSON："""


REDUCE_PROMPT = """以下是同一篇論文不同段落的線索抽取結果，請合併為最終的論文卡。

**合併規則**：
- 去重複、合併同義；當多個段落線索重疊時取最具體的
- 數字 / 指標優先採納（例：F1=0.801 比 "顯著提升" 更有用）
- title 取看起來最像論文標題的那個（通常出現在第一個 chunk）
- 摘要欄位寫 1-3 句濃縮，不要直接列線索
- 不要編造線索裡沒有的內容；對應線索全空就填 "未提及" 或空陣列

**輸出格式**：純 JSON，不要 markdown code fence、不要前後說明文字。

最終 schema：
{{
  "title": "論文標題",
  "problem": "1-2 句",
  "method": "2-3 句",
  "datasets": ["資料集"],
  "key_findings": "2-3 句，含關鍵指標",
  "contributions": ["貢獻 1", "貢獻 2", "貢獻 3"],
  "limitations": ["限制 1", "限制 2"],
  "applicable_for": "1-2 句"
}}

=== 各段落線索 ===
{partials_text}
=== END 線索 ===

請現在輸出最終 JSON："""


# === Helpers ===

def _extract_json(raw: str) -> dict:
    """從 LLM 回應抽出 JSON。容錯處理 markdown fence 與前後雜訊。"""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"找不到 JSON 物件，原始回應：\n{raw[:500]}")
    return json.loads(text[start : end + 1])


def _chunk_pages(
    pages: List[Tuple[int, str]],
    max_chars: int = MAP_CHUNK_CHARS,
) -> List[List[Tuple[int, str]]]:
    """把 [(page, text), ...] 依字數打包成多個 chunk，盡量不切斷單頁。"""
    chunks: List[List[Tuple[int, str]]] = []
    current: List[Tuple[int, str]] = []
    current_chars = 0
    for page_num, text in pages:
        if current_chars + len(text) > max_chars and current:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append((page_num, text))
        current_chars += len(text)
    if current:
        chunks.append(current)
    return chunks


# === Build paths ===

def _build_card_stuff(
    pages: List[Tuple[int, str]],
    doc_id: str,
    llm: Ollama,
) -> dict:
    full_text = "\n\n".join(text for _, text in pages)
    print(f"[build_cards] {doc_id}：stuff 模式（{len(full_text)} 字）...")
    response = llm.complete(STUFF_PROMPT.format(paper_text=full_text))
    raw = str(response)
    try:
        return _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"[build_cards] {doc_id}：JSON 解析失敗 → {e}")
        return {"_error": str(e), "_raw": raw}


def _build_card_mapreduce(
    pages: List[Tuple[int, str]],
    doc_id: str,
    llm: Ollama,
) -> dict:
    chunks = _chunk_pages(pages)
    print(
        f"[build_cards] {doc_id}：map-reduce 模式 "
        f"（{sum(len(t) for _, t in pages)} 字 → {len(chunks)} chunks）"
    )

    partials: List[dict] = []
    for idx, chunk in enumerate(chunks, start=1):
        first_page = chunk[0][0]
        last_page = chunk[-1][0]
        chunk_text = "\n\n".join(f"[p.{p}]\n{t}" for p, t in chunk)
        prompt = MAP_PROMPT.format(
            first_page=first_page,
            last_page=last_page,
            chunk_text=chunk_text,
        )
        print(
            f"[build_cards] {doc_id}：map {idx}/{len(chunks)} "
            f"(p.{first_page}–p.{last_page}, {len(chunk_text)} 字)..."
        )
        raw = str(llm.complete(prompt))
        try:
            partial = _extract_json(raw)
            partial["_chunk_pages"] = [first_page, last_page]
            partials.append(partial)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[build_cards] {doc_id}：map {idx} 解析失敗 → {e}（略過此 chunk）")

    if not partials:
        return {"_error": "all map calls failed"}

    partials_text = "\n\n".join(
        f"--- 段落 {i+1} (p.{p['_chunk_pages'][0]}–p.{p['_chunk_pages'][1]}) ---\n"
        f"{json.dumps({k: v for k, v in p.items() if k != '_chunk_pages'}, ensure_ascii=False, indent=2)}"
        for i, p in enumerate(partials)
    )
    print(f"[build_cards] {doc_id}：reduce ({len(partials)} 份線索, {len(partials_text)} 字)...")
    raw = str(llm.complete(REDUCE_PROMPT.format(partials_text=partials_text)))
    try:
        return _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"[build_cards] {doc_id}：reduce 解析失敗 → {e}")
        return {"_error": str(e), "_raw": raw, "_partials": partials}


def build_card_for_paper(
    pages: List[Tuple[int, str]],
    doc_id: str,
    llm: Ollama,
) -> dict:
    full_chars = sum(len(t) for _, t in pages)
    if full_chars <= STUFF_LIMIT_CHARS:
        card = _build_card_stuff(pages, doc_id, llm)
    else:
        card = _build_card_mapreduce(pages, doc_id, llm)
    card["doc_id"] = doc_id
    card["pages_used"] = [p for p, _ in pages]
    return card


# === CLI ===

def main(data_dir: str, force: bool, model: str) -> None:
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"資料夾不存在：{data_dir}")

    cards_dir = data_path / CARD_DIR_NAME
    cards_dir.mkdir(exist_ok=True)

    pdfs = sorted(data_path.glob("*.pdf"))
    if not pdfs:
        print(f"[build_cards] {data_dir} 下找不到 PDF。")
        return

    llm = Ollama(model=model, request_timeout=600.0)

    for pdf_path in pdfs:
        doc_id = pdf_path.stem
        out_path = cards_dir / f"{doc_id}.json"
        if out_path.exists() and not force:
            print(f"[build_cards] {doc_id}：已有卡片 → 跳過（--force 可重建）")
            continue

        print(f"\n[build_cards] === 處理 {pdf_path.name} ===")
        pages = load_pdf_pages(pdf_path)
        if not pages:
            print(f"[build_cards] {doc_id}：沒有可用頁面，跳過")
            continue

        card = build_card_for_paper(pages, doc_id, llm)
        out_path.write_text(
            json.dumps(card, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[build_cards] {doc_id}：已寫入 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Paper Cards offline")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="重建所有卡片（預設已存在的會跳過）",
    )
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    args = parser.parse_args()

    main(data_dir=args.data_dir, force=args.force, model=args.model)
