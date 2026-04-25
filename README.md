# Paper RAG with Guardrails - 期末專案

論文檢索系統，使用業界主流框架 + 開源模型，搭配雙閘 Guardrails 進行對照實驗。

## 系統架構

```
使用者查詢
    ↓
┌──────────────────────────────────────────────┐
│  Input Guardrail（NVIDIA NeMo Guardrails）    │
│  · LLM-as-judge 偵測 prompt injection         │
│  · LLM-as-judge 偵測 denied topics（離題）    │
└──────────────────────────────────────────────┘
    ↓ (pass)
┌──────────────────────────────────────────────┐
│  RAG Core（LlamaIndex）                       │
│  · Embedding：BGE-M3（Ollama 本地）           │
│  · Vector DB：Qdrant（local Docker）          │
│  · LLM：Qwen2.5:14b（Ollama 本地）            │
└──────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│  Output Guardrail（Microsoft Presidio）       │
│  · spaCy zh_core_web_lg 中文 NER              │
│  · 內建 EMAIL / PHONE 偵測                    │
│  · 自訂台灣 PII recognizer（身分證/手機/學號）│
└──────────────────────────────────────────────┘
    ↓
回傳結果
```

**所有元件都在本地執行**，無需 API key、無需網路（除首次 pull 模型）。

## Tech Stack

| 層級 | 選用 | 為什麼 |
|---|---|---|
| LLM | **Qwen2.5:14b** via Ollama | 阿里出，繁中能力最強的 14b 開源模型，M1 Max 跑起來流暢 |
| Embedding | **BGE-M3** via Ollama | 智源出，多語 embedding SOTA，中文檢索精準 |
| RAG framework | **LlamaIndex** | RAG 專業框架，比 LangChain 抽象更輕、預設行為更好 |
| Vector DB | **Qdrant** (local Docker) | Rust 寫的、性能好、現在最熱門的開源向量庫 |
| Input Guard | **NVIDIA NeMo Guardrails** | 業界主流，用 Colang DSL + LLM-as-judge，不靠 regex |
| Output Guard | **Microsoft Presidio** | PII 偵測業界標準，內建多語 NER + 可擴充 recognizer |

## 安裝

需要 Python 3.10+、Docker、Ollama。

### 1. Ollama models（首次 ~10GB 下載）

```bash
ollama pull qwen2.5:14b
ollama pull bge-m3
```

### 2. Qdrant（用 Docker）

```bash
docker run -d --name qdrant-rag -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 3. Python deps

```bash
pip install -r requirements.txt
python -m spacy download zh_core_web_lg
python -m spacy download en_core_web_sm
```

**macOS 注意**：若 `nemoguardrails` 安裝時 `annoy` 編譯失敗（"cerrno not found"），用：

```bash
CPLUS_INCLUDE_PATH="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1" \
  ARCHFLAGS="-arch arm64" \
  pip install -r requirements.txt
```

## 執行

### 1. 主對照實驗（5 類測試 × 20 筆查詢）

```bash
python3 run_eval.py
```

第一次跑會：
- 建 Qdrant 索引（~30 秒）
- 載入 Qwen2.5:14b（~10 秒）
- 載入 NeMo + Presidio + spaCy 中文模型（~5 秒）
- 跑 20 筆查詢（~10 分鐘，因為每筆要打 LLM 多次）

輸出：
- 終端：每筆查詢的 baseline / guarded 對照
- `results.md`：可放進報告附錄的 Markdown 對照表
- `metrics.json`：各類別正確率

### 2. 個別模組單獨測試

```bash
python3 rag_core.py                       # RAG 端到端
python3 -m guardrails.presidio_output     # PII 遮罩示範
python3 -m guardrails.nemo_input          # input rails 示範
python3 pipeline.py                       # baseline vs guarded
```

### 3. 互動式 / 對話式 demo

```bash
python3 compare.py                  # 互動式輸入 query 看 baseline vs guarded
python3 compare.py "你的問題"       # 單發
python3 demo_chat.py                # 6 個精選 query 的對話式 demo
python3 demo_synthetic_pii.py       # 合成 PII corpus 的最壞場景 demo
```

## 檔案結構

```
paper_rag_demo/
├── README.md
├── requirements.txt
├── data/                          ← 放論文，支援 .txt / .md / .pdf
│   └── paper.pdf                  ← 待索引的論文（可放任何 PDF/TXT/MD）
├── rag_core.py                    ← LlamaIndex + Qdrant + Ollama 整合
├── pipeline.py                    ← BaselineRAG / GuardedRAG
├── test_cases.py                  ← 5 類 × 4 筆測試查詢
├── run_eval.py                    ← 主 runner，產出 results.md / metrics.json
├── compare.py                     ← 互動式對照工具
├── demo_chat.py                   ← 6 個精選 query 對話式 demo
├── synthetic_corpus.py            ← 合成 PII fixture（最壞場景）
├── demo_synthetic_pii.py          ← 最壞場景演示
└── guardrails/
    ├── __init__.py
    ├── interface.py               ← GuardrailClient ABC
    ├── nemo_input.py              ← NVIDIA NeMo Guardrails 實作
    ├── presidio_output.py         ← Microsoft Presidio 實作
    └── nemo_config/
        └── config.yml             ← NeMo Colang + LLM 設定
```

## 對照實驗設計

每筆測試查詢分 5 類：

| 類別 | 數量 | 期望 baseline | 期望 guarded |
|---|---|---|---|
| ACADEMIC_NORMAL | 4 | 正常回答 | 正常回答（不誤擋） |
| PII_EXTRACTION | 4 | 可能洩漏個資 | OUTPUT 端遮罩 |
| OFF_TOPIC | 4 | 嘗試亂答 | INPUT 端攔下 |
| PROMPT_INJECTION | 4 | 可能中招 | INPUT 端攔下 |
| BOUNDARY | 4 | 正常回答 | 正常通過（測 false positive） |

**guardrails 是唯一的實驗變數**，兩條 pipeline 共用同一個 RAGCore。

## 對應到報告章節

- **第三章 系統設計** → 上方架構圖 + `pipeline.py`
- **第四章 實驗設計** → `test_cases.py` 的 5 類分類
- **第五章 結果分析** → `results.md`（自動產出）+ `metrics.json`
- **第六章 討論** → 下方「已知限制」段落

## 已知限制（報告討論點）

1. **本地 LLM 的非確定性**：Qwen2.5 預設 temperature > 0，相同 query 多次回答有差異。
   實驗時若要嚴格重現，可在 `rag_core.py` 設 `Ollama(..., temperature=0)`。

2. **NeMo self_check_input 仍是 LLM 判斷**：對「精心設計的 jailbreak」可能仍會中招，
   但比 regex 強很多。Production 可疊加 Llama Guard 或 ProtectAI 等專門模型。

3. **Presidio 中文 NER**：`zh_core_web_lg` 對人名/地名能辨識，但對「中文姓名 + 職稱」這種組合
   召回率有限。台灣場景已用自訂 PatternRecognizer 補強身分證/手機/學號。

4. **單篇論文 corpus**：目前只索引一篇。多篇論文時，需考慮 metadata filter
   （`vector_store.add()` 加 `doc_id`）以區分來源。

## 進一步擴充方向

- [ ] Latency / cost 量測欄位加進 `PipelineResult`（雙閘 guardrails 開銷量化）
- [ ] Ablation study：拿掉單一 guardrail 看洩漏率/誤擋率變化
- [ ] 改成 Agentic：讓 retrieval 失敗時 agent 自主換 query 或換策略
- [ ] Citation-aware：每句後標 chunk_id，方便驗證 grounding
- [ ] RAGAS 評估：faithfulness / answer relevance / context precision

## 從原 mock prototype 升級的對照

| 項目 | v1 prototype | v2（現在） |
|---|---|---|
| LLM | 規則式 mock | Qwen2.5:14b（real Ollama） |
| Chunking | 自寫章節切分 | LlamaIndex SentenceSplitter |
| Retriever | BM25 + jieba | BGE-M3 + Qdrant 向量檢索 |
| Input Guard | 自寫 regex | NeMo Guardrails (LLM-as-judge) |
| Output Guard | 自寫 regex | Presidio + spaCy zh NER |
| Vector DB | 無（記憶體 BM25） | Qdrant（持久化） |
