# Paper RAG with Guardrails - 期末專案

論文檢索系統，使用業界主流框架 + 開源模型，搭配雙閘 Guardrails 進行對照實驗。

## 系統架構

```
                  使用者查詢
                      │
        ┌─────────────▼──────────────┐
        │  NeMo Input Guard           │  prompt injection / 離題
        └─────────────┬──────────────┘
                      │ (pass)
        ┌─────────────▼──────────────┐
        │  Router Agent (Stage B)     │  LLM 分類為 4 個 intent
        └──┬───────┬───────┬───────┬─┘
           │       │       │       │
       summary    qa    recommend  out_of_scope
           │       │       │       │
           ▼       ▼       ▼       ▼
       Paper    RAG +   Recommender 拒答
       Card    Qdrant   Agent
      (離線)   filter  (讀全部 cards
              by         排序+理由)
            paper_id
           │       │       │       │
           └───────┴───┬───┴───────┘
                       │
        ┌──────────────▼─────────────┐
        │ Presidio Output Guard       │  PII 兜底
        └──────────────┬─────────────┘
                       ▼
                   回傳結果

  [離線] build_cards.py → data/cards/<doc_id>.json
         map-reduce 全文 → 結構化論文卡
```

**所有元件都在本地執行**，無需 API key、無需網路（除首次 pull 模型）。

### Agent 架構（Stage B）

| Agent | 決定什麼 | 檔案 |
|---|---|---|
| **Router** | 從 4 個 action（summary / qa / recommend / out_of_scope）挑一個 + 哪幾篇論文 | `agents.py` |
| **Recommender** | 跨論文研究方向問題 → 排序候選論文 + 引用 card 欄位的推薦理由 | `agents.py` |

兩個 agent 都跑本地 Qwen2.5:14b，透過 JSON 輸出做結構化決策。Summary 類 query 不再走 RAG 切片，改吐預先 build 好的論文卡（解決 RAG 切片無法 summary 的本質問題）。

## Tech Stack

| 層級 | 選用 | 為什麼 |
|---|---|---|
| LLM | **Qwen2.5:14b** via Ollama | 阿里出，繁中能力最強的 14b 開源模型，M1 Max 跑起來流暢 |
| Embedding | **BGE-M3** via Ollama | 智源出，多語 embedding SOTA，中文檢索精準 |
| RAG framework | **LlamaIndex** | RAG 專業框架，比 LangChain 抽象更輕、預設行為更好 |
| Vector DB | **Qdrant** (local Docker) | Rust 寫的、性能好、metadata filter 強（多論文場景能用 `paper_id` 隔離） |
| Input Guard | **NVIDIA NeMo Guardrails** | 業界主流，用 Colang DSL + LLM-as-judge，不靠 regex |
| Output Guard | **Microsoft Presidio** | PII 偵測業界標準，內建多語 NER + 可擴充 recognizer |
| Agent layer | **Router + Recommender** (自寫於 `agents.py`) | 用本地 LLM JSON-mode 做結構化決策；Stage B 把固定 if/else 升級成 agent-driven dispatch |

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

### 2.5 為新論文 build 論文卡（Stage A 必要、Stage B Recommender 仰賴）

```bash
python3 build_cards.py                    # 為 data/*.pdf 全部產卡（已存在會跳過）
python3 build_cards.py --force            # 強制重建所有卡
```

長論文走 map-reduce（每段抽事實線索 → 合併為最終 card），輸出落到 `data/cards/<doc_id>.json`。
Router 命中 `summary` 或 `recommend` intent 時會直接讀這些卡，不走 RAG 切片。

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
├── report.md                      ← 完整報告（給老師）
├── requirements.txt
├── data/                          ← 放論文，支援 .txt / .md / .pdf
│   ├── paper.pdf                  ← 待索引的論文
│   └── cards/                     ← 離線 build 的論文卡（JSON）
├── rag_core.py                    ← LlamaIndex + Qdrant + Ollama 整合（含 TOC 過濾、metadata filter）
├── pipeline.py                    ← BaselineRAG / GuardedRAG（Stage B：Router 分流）
├── agents.py                      ← Router + Recommender agents
├── build_cards.py                 ← 離線論文卡產生器（map-reduce）
├── test_cases.py                  ← 5 類 × 4 筆測試查詢
├── run_eval.py                    ← 主 runner，產出 results.md / metrics.json
├── compare.py                     ← 互動式對照工具
├── demo_chat.py                   ← 6 個精選 query 對話式 demo
├── synthetic_corpus.py            ← 合成 PII fixture（最壞場景）
├── demo_synthetic_pii.py          ← 最壞場景演示
├── app.py                         ← FastAPI 後端（/api/chat、Swagger UI）
├── static/index.html              ← ChatGPT 風格對話前端
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

4. **多論文 corpus 尚未實測**：infrastructure 已就緒（`paper_id` metadata filter、Router、Recommender 全部支援多論文），但 `data/` 目前只有一篇 PDF；正式跨論文 demo 需先補論文 + 跑 `build_cards.py` + `--rebuild` 索引。

5. **Router 也是 LLM 判斷**：與 NeMo 同樣有「自己審自己」的盲點；Critic agent（反幻覺檢查）尚未實作，這是 Stage C 的工作。

## 進一步擴充方向

- [ ] Latency / cost 量測欄位加進 `PipelineResult`（雙閘 guardrails 開銷量化）
- [ ] Ablation study：拿掉單一 guardrail 看洩漏率/誤擋率變化
- [ ] **Stage C：Critic agent**（反幻覺）— 對 qa / recommend 回應做事後逐句檢驗
- [ ] **Stage C：Per-paper Retriever agent**（跨論文平行檢索）— 推薦時對每篇候選論文獨立做 RAG，把實證 chunks 餵給 Recommender
- [ ] Citation-aware：每句後標 chunk_id，方便驗證 grounding
- [ ] RAGAS 評估：faithfulness / answer relevance / context precision

## 從原 mock prototype 升級的對照

| 項目 | v1 prototype | v2（雙閘 guardrails） | v3（agentic，現在） |
|---|---|---|---|
| LLM | 規則式 mock | Qwen2.5:14b（real Ollama） | 同 v2 |
| Chunking | 自寫章節切分 | LlamaIndex SentenceSplitter | + TOC 頁過濾 |
| Retriever | BM25 + jieba | BGE-M3 + Qdrant 向量檢索 | + `paper_id` metadata filter |
| Input Guard | 自寫 regex | NeMo Guardrails (LLM-as-judge) | 同 v2 |
| Output Guard | 自寫 regex | Presidio + spaCy zh NER | 同 v2 |
| Vector DB | 無（記憶體 BM25） | Qdrant（持久化） | 同 v2 |
| Query 分流 | — | 寫死的 if/else | **Router agent**（LLM 分 4 intent） |
| Summary | RAG 切片硬湊（不 work） | 同 v1 | **Paper Card** map-reduce 全文摘要 |
| 跨論文推薦 | — | — | **Recommender agent**（讀 cards 排序） |
