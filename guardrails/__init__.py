"""Guardrails 模組

兩個實作都對齊 GuardrailClient 介面：
  - NemoInputGuard：NVIDIA NeMo Guardrails，負責 INPUT 端攻擊偵測
  - PresidioOutputGuard：Microsoft Presidio，負責 OUTPUT 端 PII 遮罩

pipeline.py 預設配置：INPUT 用 Nemo + OUTPUT 用 Presidio
"""
from .interface import GuardrailClient, GuardrailResponse
from .presidio_output import PresidioOutputGuard
from .nemo_input import NemoInputGuard

__all__ = [
    "GuardrailClient",
    "GuardrailResponse",
    "PresidioOutputGuard",
    "NemoInputGuard",
]
