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
    idioma_origem:  str = "en"
    idioma_destino: str = "pt"

    # ── Whisper ───────────────────────────────────────────────────────────────
    # tiny | small | medium (recomendado) | large (mais preciso, lento)
    whisper_model: str  = "medium"

    # ── Coqui XTTS-v2 ─────────────────────────────────────────────────────────
    # True  → clona a voz extraída do próprio vídeo (recomendado)
    # False → usa voz padrão multilíngue do XTTS
    clonar_voz:    bool  = True

    # Duração do trecho inicial usado como referência de fallback (mínimo 3s).
    # Usado apenas se a diarização não estiver disponível.
    ref_duracao_s: float = 10.0

    # ── Diarização de falantes (100% offline) ─────────────────────────────────
    # True  → usa resemblyzer + KMeans para identificar e separar falantes
    # False → usa a mesma voz de referência para todos os segmentos
    # Requer: pip install resemblyzer scikit-learn soundfile
    diarizar:      bool  = True

    # Número máximo de falantes esperados no vídeo.
    # O algoritmo estima automaticamente; este é o teto.
    max_falantes:  int   = 6

    # ── Sincronização ─────────────────────────────────────────────────────────
    # "stretch" → ajusta velocidade do áudio (sem alterar pitch) para caber no slot
    # "fill"    → corta ou preenche com silêncio
    modo_sync:     str   = "stretch"

    # Segmentos com duração < min_seg_s são mesclados ao próximo.
    min_seg_s:     float = 1.5

    # ── Hardware ──────────────────────────────────────────────────────────────
    usar_gpu:      bool  = torch.cuda.is_available()

    # ── Exportação ────────────────────────────────────────────────────────────
    video_codec:   str = "libx264"
    audio_codec:   str = "aac"
    audio_bitrate: str = "192k"
    crf:           int = 18