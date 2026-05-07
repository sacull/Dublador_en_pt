"""
config.py
─────────
Todas as configurações do projeto em um único lugar.
Edite este arquivo antes de rodar o main.py.
"""

import torch
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
# MAPEAMENTO DE IDIOMAS  Argos Translate → Coqui XTTS-v2
# ══════════════════════════════════════════════════════════════════════════════
XTTS_LANG: dict[str, str] = {
    "pt": "pt", "en": "en", "es": "es",
    "fr": "fr", "de": "de", "it": "it",
    "ja": "ja", "zh": "zh-cn", "pl": "pl",
    "nl": "nl", "ru": "ru",   "tr": "tr",
    "ko": "ko", "ar": "ar",   "cs": "cs",
    "hu": "hu", "ro": "ro",
}


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES GERAIS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:

    # ── Arquivos ──────────────────────────────────────────────────────────────
    video_entrada: str  = "video.mp4"
    video_saida:   str  = "video_dublado.mp4"

    # ── Idiomas ───────────────────────────────────────────────────────────────
    # Códigos Argos: en, es, fr, de, it, ja, zh, ru, pl, nl, ar, ko, pt…
    idioma_origem:  str = "en"
    idioma_destino: str = "pt"   # pt = português (Argos não diferencia BR/PT)

    # ── Whisper ───────────────────────────────────────────────────────────────
    # tiny (rápido) | small | medium (recomendado) | large (mais preciso, lento)
    whisper_model: str  = "medium"

    # ── Coqui XTTS-v2 ─────────────────────────────────────────────────────────
    # True  → clona a voz extraída do próprio vídeo (recomendado)
    # False → usa voz padrão multilíngue do XTTS
    clonar_voz:    bool  = True

    # Duração do trecho inicial usado como referência de clonagem (mínimo 3s).
    # Aumente para 15–20s se o início do vídeo tiver ruído ou música.
    ref_duracao_s: float = 10.0

    # ── Sincronização ─────────────────────────────────────────────────────────
    # "stretch" → ajusta velocidade do áudio (sem alterar pitch) para caber no slot
    # "fill"    → corta ou preenche com silêncio
    modo_sync:     str   = "stretch"

    # Segmentos com duração < min_seg_s são mesclados ao próximo.
    # O XTTS gera áudio ruim para frases muito curtas (1–2 palavras).
    min_seg_s:     float = 1.5

    # ── Hardware ──────────────────────────────────────────────────────────────
    # auto-detecta GPU NVIDIA; force False para rodar só em CPU
    usar_gpu:      bool  = torch.cuda.is_available()

    # ── Exportação ────────────────────────────────────────────────────────────
    video_codec:   str = "libx264"
    audio_codec:   str = "aac"
    audio_bitrate: str = "192k"
    crf:           int = 18    # 0=sem perda · 18=alta qualidade · 28=menor arquivo