"""
Guardrails 介面層 - 統一所有實作的呼叫方式

回傳格式遵循 AWS Bedrock ApplyGuardrail API 的 response schema，
這樣 pipeline.py 不論換哪個 guardrail 實作都不用改程式。
"""
from abc import ABC, abstractmethod
from typing import Literal, TypedDict, List


class GuardrailContent(TypedDict):
    """單一文字輸入片段"""
    text: dict  # {"text": "..."}


class GuardrailAssessment(TypedDict, total=False):
    """評估細節 - 對應 Bedrock 各 policy"""
    topicPolicy: dict
    contentPolicy: dict
    sensitiveInformationPolicy: dict
    wordPolicy: dict
    contextualGroundingPolicy: dict


class GuardrailResponse(TypedDict):
    """ApplyGuardrail API 回傳結構（簡化版）"""
    action: Literal["GUARDRAIL_INTERVENED", "NONE"]
    output: List[dict]  # 通過時為原文，遮罩時為遮罩版，封鎖時為固定訊息
    assessments: List[GuardrailAssessment]


class GuardrailClient(ABC):
    """所有 Guardrail 實作的共同介面"""

    @abstractmethod
    def apply_guardrail(
        self,
        source: Literal["INPUT", "OUTPUT"],
        text: str,
    ) -> GuardrailResponse:
        """
        評估文字內容
        - source="INPUT"  代表使用者輸入（攻擊偵測會更嚴格）
        - source="OUTPUT" 代表 LLM 生成內容（PII 遮罩、幻覺檢查重點）
        """
        ...
