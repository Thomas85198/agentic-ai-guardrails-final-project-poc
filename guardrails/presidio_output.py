"""
Presidio Output Guardrail - PII 偵測與遮罩

技術棧：
  - Microsoft Presidio Analyzer / Anonymizer（業界標準）
  - spaCy 中文 NER 模型（zh_core_web_lg）做基礎實體辨識
  - 自訂 PatternRecognizer 處理台灣特有 PII（身分證、手機、學號）

設計重點：
  - PERSON 預設 ALLOW（學術論文作者/教授名為公開資訊，誤擋成本高）
  - EMAIL / PHONE_NUMBER / 台灣 ID 預設 ANONYMIZE
  - 可由建構參數調整 pii_action（BLOCK / ANONYMIZE）
"""
from typing import Literal

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .interface import GuardrailClient, GuardrailResponse


# === 台灣自訂 PII recognizers（用 Presidio API，非 raw regex 用法）===
TW_NATIONAL_ID = PatternRecognizer(
    supported_entity="TW_NATIONAL_ID",
    name="TW National ID",
    patterns=[Pattern(name="tw_id", regex=r"\b[A-Z][12]\d{8}\b", score=0.95)],
    supported_language="zh",
)

TW_MOBILE = PatternRecognizer(
    supported_entity="TW_MOBILE",
    name="TW Mobile",
    patterns=[Pattern(name="tw_mobile", regex=r"09\d{2}[- ]?\d{3}[- ]?\d{3}", score=0.9)],
    supported_language="zh",
)

TW_STUDENT_ID = PatternRecognizer(
    supported_entity="TW_STUDENT_ID",
    name="TW Student ID",
    patterns=[Pattern(name="tw_sid", regex=r"\b1\d{8}\b", score=0.85)],
    supported_language="zh",
)

# 預設關注的 PII 類別
# 不含 URL（學術論文常引用 URL，誤擋成本高）
# 不含 PERSON（論文作者/教授名為公開學術資訊，誤擋成本更高）
DEFAULT_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "TW_NATIONAL_ID",
    "TW_MOBILE",
    "TW_STUDENT_ID",
]


def _build_analyzer() -> AnalyzerEngine:
    """建立支援中英雙語的 Analyzer，並掛上台灣自訂 recognizer"""
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "zh", "model_name": "zh_core_web_lg"},
            {"lang_code": "en", "model_name": "en_core_web_sm"},
        ],
    }
    # 若 en_core_web_sm 沒裝，退回單語（中文場景仍可運作）
    try:
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["zh", "en"],
        )
    except Exception:
        # fallback: 純中文
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "zh", "model_name": "zh_core_web_lg"}],
        })
        analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=["zh"],
        )
    # 掛上台灣 recognizers
    for r in (TW_NATIONAL_ID, TW_MOBILE, TW_STUDENT_ID):
        analyzer.registry.add_recognizer(r)
    return analyzer


class PresidioOutputGuard(GuardrailClient):
    """以 Presidio 為核心的 OUTPUT guardrail（INPUT 直接 pass-through）"""

    def __init__(
        self,
        pii_action: Literal["BLOCK", "ANONYMIZE"] = "ANONYMIZE",
        entities: list = None,
        language: str = "zh",
    ):
        self.analyzer = _build_analyzer()
        self.anonymizer = AnonymizerEngine()
        self.pii_action = pii_action
        self.entities = entities or DEFAULT_ENTITIES
        self.language = language

    def apply_guardrail(
        self, source: Literal["INPUT", "OUTPUT"], text: str
    ) -> GuardrailResponse:
        # v2 雙向 PII 防禦（縱深設計）：INPUT / OUTPUT 兩端都跑 Presidio。
        # 設計原則：
        #   - INPUT 端遮罩 PII → 確保 LLM context、log、retrieval 都看不到原始 PII
        #   - OUTPUT 端再過一次 → 兜底「LLM 從 chunks 撈到 PII」的情況
        # source 仍保留以利 assessment 標示是哪一端觸發。

        # 跑 Presidio Analyzer
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language=self.language,
        )

        if not results:
            return {"action": "NONE", "output": [{"text": text}], "assessments": []}

        if self.pii_action == "BLOCK":
            output_text = "Sorry, the response contains sensitive information and was blocked."
        else:
            # 用 Presidio Anonymizer 做標準遮罩
            operators = {
                "DEFAULT": OperatorConfig(
                    "replace",
                    {"new_value": "[REDACTED]"},
                ),
                # 不同類型可不同 placeholder，便於肉眼閱讀
                "EMAIL_ADDRESS":   OperatorConfig("replace", {"new_value": "[EMAIL_REDACTED]"}),
                "PHONE_NUMBER":    OperatorConfig("replace", {"new_value": "[PHONE_REDACTED]"}),
                "TW_NATIONAL_ID":  OperatorConfig("replace", {"new_value": "[TW_NATIONAL_ID_REDACTED]"}),
                "TW_MOBILE":       OperatorConfig("replace", {"new_value": "[TW_MOBILE_REDACTED]"}),
                "TW_STUDENT_ID":   OperatorConfig("replace", {"new_value": "[TW_STUDENT_ID_REDACTED]"}),
            }
            anonymized = self.anonymizer.anonymize(
                text=text, analyzer_results=results, operators=operators,
            )
            output_text = anonymized.text

        assessment = {
            "sensitiveInformationPolicy": {
                "source": source,
                "piiEntities": [
                    {
                        "type": r.entity_type,
                        "match": text[r.start:r.end],
                        "score": r.score,
                        "action": self.pii_action,
                    }
                    for r in results
                ]
            }
        }
        return {
            "action": "GUARDRAIL_INTERVENED",
            "output": [{"text": output_text}],
            "assessments": [assessment],
        }


if __name__ == "__main__":
    g = PresidioOutputGuard()
    samples = [
        "正常學術內容，無 PII 應放行：本研究方法的 F1 為 0.801。",
        "聯絡作者請寄信到 test.author@example.com 或致電 0912-345-678，學號 109522028。",
        "受試者 A 身分證 A123456789，居住於台北市信義區。",
    ]
    for s in samples:
        print(f"\n>>> {s}")
        r = g.apply_guardrail("OUTPUT", s)
        print(f"action: {r['action']}")
        print(f"output: {r['output'][0]['text']}")
        if r["assessments"]:
            for e in r["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"]:
                print(f"  - {e['type']}: {e['match']} (score={e['score']:.2f})")
