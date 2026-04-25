"""
RAG Core - LlamaIndex + Qdrant + Ollama 的整合層

設計：
  - Embedding：BGE-M3（Ollama 本地）
  - Vector DB：Qdrant（local Docker on :6333）
  - LLM：Qwen2.5:14b（Ollama 本地）
  - Chunking：LlamaIndex SentenceSplitter，500 tokens / 50 overlap

文件來源：
  data/ 資料夾下所有 .txt / .md / .pdf，每個檔案 → 一個 Document
  （doc_id 取檔名 stem，方便日後溯源）

第一次跑會建 collection 並索引 data/ 下所有文件；之後重跑會直接 reuse。
新增/更換文件後要重建索引：
  - 程式內：RAGCore(force_rebuild=True)
  - 命令列：python rag_core.py --rebuild
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pypdf import PdfReader
from qdrant_client import QdrantClient


# === 預設設定（可在 build_rag 覆寫）===
DEFAULT_LLM_MODEL = "qwen2.5:14b"
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_COLLECTION = "paper_rag"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_DATA_DIR = "data"
SUPPORTED_EXTS = {".txt", ".md", ".pdf"}

# TOC 偵測門檻：dot ratio 來自「去空白後 . 的占比」，目錄頁通常 > 30%
_TOC_DOT_RATIO = 0.15
# leader-dots 出現次數（5 個以上連續點，或被空白隔開的 . . . . . 也算）
_TOC_LEADER_DOTS_MIN = 5
_LEADER_DOTS_RE = re.compile(r"(?:\.\s*){5,}")


def _is_toc_page(text: str) -> tuple[bool, float, int]:
    """偵測 PDF 頁是否為目錄/圖目錄/表目錄。回傳 (是否跳過, dot_ratio, leader 次數)。"""
    non_ws = "".join(text.split())
    if len(non_ws) < 50:
        return False, 0.0, 0
    dot_ratio = non_ws.count(".") / len(non_ws)
    leader_count = len(_LEADER_DOTS_RE.findall(text))
    is_toc = dot_ratio > _TOC_DOT_RATIO or leader_count >= _TOC_LEADER_DOTS_MIN
    return is_toc, dot_ratio, leader_count


def load_pdf_pages(path: Path, verbose: bool = True) -> List[tuple[int, str]]:
    """載入 PDF,跳過目錄頁,回傳 [(page_num, text), ...]。給 RAGCore 與 build_cards 共用。"""
    reader = PdfReader(str(path))
    pages: List[tuple[int, str]] = []
    skipped: List[int] = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue
        is_toc, dot_ratio, leader = _is_toc_page(page_text)
        if is_toc:
            skipped.append(i)
            if verbose:
                print(
                    f"[load_pdf_pages] 跳過目錄頁：{path.name} p.{i} "
                    f"(dot_ratio={dot_ratio:.2f}, leaders={leader})"
                )
            continue
        pages.append((i, page_text))
    if verbose:
        msg = f"[load_pdf_pages] 載入：{path.name}（{len(pages)} 頁）"
        if skipped:
            msg += f"，跳過目錄 {len(skipped)} 頁：{skipped}"
        print(msg)
    return pages


@dataclass
class RetrievedChunk:
    """檢索結果，給 pipeline 與 evaluator 用"""
    chunk_id: str
    text: str
    score: float
    doc_id: str = ""
    page: Optional[int] = None  # PDF 才有；txt/md 為 None


class RAGCore:
    """LlamaIndex 包裝；對外露出 retrieve() 與 query()"""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        llm_model: str = DEFAULT_LLM_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        top_k: int = 5,
        force_rebuild: bool = False,
        documents: Optional[List[Document]] = None,
    ):
        # 全域 LlamaIndex 設定
        Settings.llm = Ollama(
            model=llm_model,
            request_timeout=120.0,
            # 提示模板用論文檢索場景的繁中指令
            system_prompt=(
                "你是一個學術論文檢索助手。"
                "請根據提供的論文片段，用繁體中文回答使用者問題。"
                "若片段未涵蓋答案，請直接說「論文中未提及」，不要編造。"
            ),
        )
        Settings.embed_model = OllamaEmbedding(model_name=embed_model)
        Settings.node_parser = SentenceSplitter(
            chunk_size=500,  # 對應約 700-900 中文字
            chunk_overlap=50,
            paragraph_separator="\n\n",
        )

        self.top_k = top_k
        self.collection_name = collection_name

        # Qdrant client
        self.qdrant = QdrantClient(url=qdrant_url)
        existing = {c.name for c in self.qdrant.get_collections().collections}

        # 注意：刪除 collection 必須在 QdrantVectorStore 建立**之前**做。
        # QdrantVectorStore.__init__ 只檢查一次 _collection_initialized，
        # 若先建立再刪，後續 add() 不會自動重建 collection，導致 404。
        if force_rebuild and collection_name in existing:
            self.qdrant.delete_collection(collection_name)
            existing.discard(collection_name)

        vector_store = QdrantVectorStore(
            client=self.qdrant,
            collection_name=collection_name,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if collection_name not in existing:
            # 首次建索引：用傳入 documents 或從 data_dir 載入
            print(f"[RAGCore] 建立新索引：{collection_name}")
            docs = documents if documents is not None else self._load_documents(data_dir)
            self.index = VectorStoreIndex.from_documents(
                docs,
                storage_context=storage_context,
                show_progress=True,
            )
        else:
            # 已存在 → 直接掛上
            print(f"[RAGCore] 載入既有索引：{collection_name}")
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
            )

        self.retriever = self.index.as_retriever(similarity_top_k=top_k)
        self.query_engine = self.index.as_query_engine(similarity_top_k=top_k)

    # === 載文件 ===
    def _load_documents(self, data_dir: str) -> List[Document]:
        """掃描 data_dir 下所有支援的檔案，產生 Document 列表。

        - .txt / .md：整檔一個 Document
        - .pdf：**每頁一個 Document**，metadata 帶 page，方便日後引用標頁碼
        所有 Document 都會帶 metadata['doc_id'] = 檔名 stem，多論文場景可以對應回來源檔。
        """
        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise FileNotFoundError(f"資料夾不存在：{data_dir}")

        docs: List[Document] = []
        for path in sorted(data_path.iterdir()):
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTS:
                continue
            doc_id = path.stem
            if suffix in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8")
                if not text.strip():
                    print(f"[RAGCore] 跳過空檔案：{path.name}")
                    continue
                print(f"[RAGCore] 載入：{path.name}（{len(text)} chars）")
                docs.append(
                    Document(
                        text=text,
                        doc_id=doc_id,
                        metadata={"doc_id": doc_id},
                    )
                )
            elif suffix == ".pdf":
                for page_num, page_text in load_pdf_pages(path):
                    docs.append(
                        Document(
                            text=page_text,
                            doc_id=f"{doc_id}#p{page_num}",
                            metadata={"doc_id": doc_id, "page": page_num},
                        )
                    )

        if not docs:
            raise FileNotFoundError(
                f"{data_dir} 中沒有可索引的文件（支援副檔名：{sorted(SUPPORTED_EXTS)}）"
            )
        return docs

    @staticmethod
    def _node_to_chunk(n: NodeWithScore) -> RetrievedChunk:
        """把 LlamaIndex NodeWithScore 抽成 RetrievedChunk，連帶帶出 doc_id/page metadata。"""
        meta = n.node.metadata or {}
        return RetrievedChunk(
            chunk_id=n.node.node_id[:12],
            text=n.node.get_content(),
            score=float(n.score or 0.0),
            doc_id=meta.get("doc_id", ""),
            page=meta.get("page"),
        )

    # === 給 pipeline 用：純檢索 ===
    def retrieve(self, query: str) -> List[RetrievedChunk]:
        nodes: List[NodeWithScore] = self.retriever.retrieve(query)
        return [self._node_to_chunk(n) for n in nodes]

    # === 給 pipeline 用：完整 RAG（檢索 + LLM 生成）===
    def query(self, query: str) -> tuple[str, List[RetrievedChunk]]:
        response = self.query_engine.query(query)
        chunks = [self._node_to_chunk(n) for n in response.source_nodes]
        return str(response), chunks

    # === 給 chat API 用：多輪對話 with in-context learning ===
    def chat(self, messages: List[dict]) -> tuple[str, List[RetrievedChunk]]:
        """
        多輪對話：把對話歷史與 retrieved context 一起餵給 LLM。

        messages 格式：[{"role": "user"|"assistant", "content": "..."}, ...]
        最後一筆必須是 user。

        設計：
          - retrieval 只用最新 user query（避免雜訊）
          - 把 retrieved context 塞進 system prompt
          - 完整 chat history 一起送給 LLM，實現 in-context learning
        """
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("messages 不能為空，且最後一筆必須是 user")

        latest_query = messages[-1]["content"]
        chunks = self.retrieve(latest_query)

        def _src_label(c: RetrievedChunk) -> str:
            """組成人類/模型都看得懂的來源標籤,例如 'paper, p.3' 或 'paper'。"""
            if not c.doc_id:
                return ""
            return f"{c.doc_id}, p.{c.page}" if c.page is not None else c.doc_id

        context_text = "\n\n".join(
            f"[chunk {i+1}] (來源: {_src_label(c) or '未知'})\n{c.text}"
            for i, c in enumerate(chunks)
        )

        system_content = (
            "你是一個學術論文檢索助手。"
            "請根據以下提供的論文片段，用繁體中文回答使用者問題。"
            "若片段未涵蓋答案，請直接說「論文中未提及」，不要編造。"
            "你可以參考前面的對話脈絡來理解使用者的追問。\n\n"
            "**引用格式（重要）**：回答時請在每個論點/事實後面用 [chunk N] 標註對應片段，"
            "例如：「本論文採用 BERT 作為基線 [chunk 1]，並在 SQuAD 上評估 [chunk 2]。」"
            "若多個片段共同支撐同一論點，可寫 [chunk 1][chunk 3]。"
            "請**只**使用上面實際提供的 chunk 編號，不要編造不存在的編號。\n\n"
            f"=== 論文片段 ===\n{context_text}\n=== END 論文片段 ==="
        )

        chat_history = [ChatMessage(role="system", content=system_content)]
        for m in messages:
            chat_history.append(
                ChatMessage(role=m["role"], content=m["content"])
            )

        response = Settings.llm.chat(chat_history)
        return str(response.message.content), chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGCore smoke test")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="刪掉現有 collection 重建索引（新增/更換 data/ 文件後使用）",
    )
    args = parser.parse_args()

    print("初始化 RAG core...")
    rag = RAGCore(force_rebuild=args.rebuild)
    print("\n=== 測試檢索 ===")
    chunks = rag.retrieve("這篇論文的主要研究方法是什麼？")
    for c in chunks:
        print(f"\n[{c.chunk_id}] score={c.score:.3f}")
        print(f"  {c.text[:120]}...")

    print("\n\n=== 測試完整 RAG ===")
    answer, _ = rag.query("請簡述這篇論文的主要貢獻。")
    print(answer)
