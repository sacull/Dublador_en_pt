"""
tradutor.py
───────────
Responsabilidade: traduzir os segmentos transcritos usando Argos Translate,
que roda 100% offline após o primeiro download do pacote de idiomas.
"""

import argostranslate.package
import argostranslate.translate
from tqdm import tqdm

from config import Config


class Tradutor:
    """
    Wrapper sobre o Argos Translate para tradução offline entre pares de idiomas.

    Na primeira execução baixa e instala o pacote do par de idiomas (~100 MB).
    Nas execuções seguintes detecta que já está instalado e não acessa a internet.
    """

    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self._tradutor = None   # inicializado com lazy loading em _obter_tradutor()

    def traduzir(self, segmentos: list[dict]) -> list[dict]:
        """
        Traduz o campo 'text' de cada segmento e adiciona 'texto_traduzido'.
        Retorna a mesma lista com o campo novo preenchido.
        """
        print(f"\n[3/6] Traduzindo {len(segmentos)} segmentos com Argos Translate (offline)…")

        self._garantir_pacote()
        tradutor = self._obter_tradutor()

        for seg in tqdm(segmentos, desc="  Traduzindo"):
            texto = seg["text"].strip()
            seg["texto_traduzido"] = tradutor.translate(texto) if texto else ""

        self._mostrar_amostra(segmentos)
        return segmentos

    # ── Métodos privados ──────────────────────────────────────────────────────

    def _garantir_pacote(self) -> None:
        """
        Verifica se o par de idiomas já está instalado localmente.
        Se não estiver, baixa e instala (requer internet apenas nesta etapa).
        """
        from_code = self.cfg.idioma_origem
        to_code   = self.cfg.idioma_destino

        instalados = argostranslate.translate.get_installed_languages()
        ja_instalado = any(
            lang.code == from_code and
            any(t.to_lang.code == to_code for t in lang.translations_from)
            for lang in instalados
        )

        if ja_instalado:
            print(f"      Pacote {from_code}→{to_code} já instalado.")
            return

        print(f"      Baixando pacote Argos {from_code}→{to_code} (~100 MB, apenas 1 vez)…")
        argostranslate.package.update_package_index()

        pacotes     = argostranslate.package.get_available_packages()
        pacote_alvo = next(
            (p for p in pacotes if p.from_code == from_code and p.to_code == to_code),
            None,
        )

        if pacote_alvo is None:
            raise RuntimeError(
                f"Par de idiomas '{from_code}→{to_code}' não encontrado no Argos.\n"
                f"Consulte os pares disponíveis em: "
                f"https://github.com/argosopentech/argos-translate"
            )

        argostranslate.package.install_from_path(pacote_alvo.download())
        print("      Pacote instalado com sucesso.")

    def _obter_tradutor(self):
        """Retorna o objeto de tradução do par de idiomas configurado (lazy load)."""
        if self._tradutor is not None:
            return self._tradutor

        instalados = argostranslate.translate.get_installed_languages()
        lang_from  = next(l for l in instalados if l.code == self.cfg.idioma_origem)
        lang_to    = next(l for l in instalados if l.code == self.cfg.idioma_destino)
        self._tradutor = lang_from.get_translation(lang_to)
        return self._tradutor

    def _mostrar_amostra(self, segmentos: list[dict]) -> None:
        """Exibe os 3 primeiros segmentos traduzidos para conferência."""
        for seg in segmentos[:3]:
            print(f"  [{seg['start']:.1f}s→{seg['end']:.1f}s]  {seg['text'][:55]}")
            print(f"   ↳ {seg['texto_traduzido'][:55]}")