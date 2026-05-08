"""
main.py
───────
Ponto de entrada do projeto. Orquestra todas as etapas da pipeline de dublagem
instanciando cada classe responsável e passando os dados entre elas.

Estrutura do projeto:
    config.py        → configurações e constantes
    extrator.py      → extração de áudio do vídeo (FFmpeg)
    transcritor.py   → transcrição com Whisper + mesclagem de segmentos curtos
    diarizador.py    → separação de falantes offline (resemblyzer + KMeans)
    tradutor.py      → tradução offline com Argos Translate
    sintetizador.py  → síntese de voz com Coqui XTTS-v2 + sincronização
    exportador.py    → montagem do vídeo final (FFmpeg)
    main.py          → orquestração da pipeline completa  ← você está aqui

Dependências obrigatórias:
    pip install openai-whisper argostranslate TTS pydub ffmpeg-python tqdm torch

Dependências opcionais (diarização de múltiplos falantes, 100% offline):
    pip install resemblyzer scikit-learn soundfile

Sistema:
    ffmpeg instalado e no PATH
    Python 3.10 ou 3.11 (obrigatório para o Coqui TTS)
"""

import sys
import subprocess
import tempfile
from pathlib import Path

# ── Auto-instalação de dependências obrigatórias ──────────────────────────────
DEPS = [
    "openai-whisper",
    "argostranslate",
    "TTS",
    "pydub",
    "ffmpeg-python",
    "tqdm",
    "torch",
]

def instalar_deps() -> None:
    for dep in DEPS:
        pkg = dep.replace("-", "_").lower()
        try:
            __import__(pkg)
        except ImportError:
            print(f"[setup] Instalando {dep}…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])

instalar_deps()

# ── Imports do projeto ────────────────────────────────────────────────────────
from config       import Config
from extrator     import Extrator
from transcritor  import Transcritor
from diarizador   import Diarizador
from tradutor     import Tradutor
from sintetizador import Sintetizador
from exportador   import Exportador


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class Pipeline:
    """
    Orquestra as etapas da dublagem:
        1.  Extração de áudio
        2.  Transcrição (Whisper)
        3.  Diarização offline (resemblyzer + KMeans) — separa falantes
        4.  Tradução (Argos Translate — offline)
        5.  Carregamento do modelo TTS (Coqui XTTS-v2)
        6.  Síntese de voz + sincronização (voz diferente por falante)
        7.  Exportação do vídeo final
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.tmp    = Path(tempfile.mkdtemp(prefix="dub_"))
        self.device = "cuda" if cfg.usar_gpu else "cpu"

        print("=" * 60)
        print("  Dublador de Vídeo — 100% Offline")
        print("=" * 60)
        print(f"  Entrada    : {cfg.video_entrada}")
        print(f"  Saída      : {cfg.video_saida}")
        print(f"  Idiomas    : {cfg.idioma_origem} → {cfg.idioma_destino}")
        print(f"  Whisper    : {cfg.whisper_model}")
        print(f"  Voz        : {'clonada do vídeo' if cfg.clonar_voz else 'padrão XTTS'}")
        print(f"  Diarização : {'ativada (resemblyzer)' if cfg.diarizar else 'desativada'}")
        print(f"  Hardware   : {self.device.upper()}")
        print(f"  Temp       : {self.tmp}")
        print("=" * 60)

    def executar(self) -> None:
        # 1. Extração
        extrator  = Extrator(self.cfg, self.tmp)
        wav       = extrator.extrair()

        # 2. Transcrição
        transcritor = Transcritor(self.cfg, self.tmp, self.device)
        segmentos   = transcritor.transcrever(wav)

        # 3. Diarização (anota 'speaker' em cada segmento e extrai referências de voz)
        if self.cfg.diarizar:
            diarizador = Diarizador(self.cfg, self.tmp)
            segmentos  = diarizador.diarizar(wav, segmentos)

        # 4. Tradução
        tradutor  = Tradutor(self.cfg)
        segmentos = tradutor.traduzir(segmentos)

        # 5 + 6. Síntese
        sintetizador = Sintetizador(self.cfg, self.tmp, self.device)
        modelo       = sintetizador.carregar_modelo()
        audio_final  = sintetizador.sintetizar(segmentos, modelo)

        # 7. Exportação
        exportador = Exportador(self.cfg)
        exportador.exportar(audio_final)


# ══════════════════════════════════════════════════════════════════════════════
# PONTO DE ENTRADA — edite as configurações aqui
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = Config(
        video_entrada  = "video.mp4",
        video_saida    = "video_dublado.mp4",
        idioma_origem  = "en",
        idioma_destino = "pt",
        whisper_model  = "large",
        clonar_voz     = True,
        ref_duracao_s  = 10.0,
        modo_sync      = "stretch",
        diarizar       = True,    # False para desativar e usar voz única
        max_falantes   = 6,
    )

    Pipeline(cfg).executar()