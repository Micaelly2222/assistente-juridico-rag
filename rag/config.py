import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CORPUS = DATA / "corpus.jsonl"
EMBEDDINGS = DATA / "embeddings.npy"
EVAL_QUESTIONS = DATA / "eval_questions.json"
REPORTS = ROOT / "reports"


@dataclass(frozen=True)
class Law:
    key: str
    name: str
    short: str
    url: str


# Texto da publicação original no portal da Câmara dos Deputados.
# Leis não têm direito autoral no Brasil (Lei 9.610/98, art. 8º, IV).
LAWS = [
    Law("lgpd", "Lei nº 13.709/2018 - Lei Geral de Proteção de Dados", "LGPD",
        "https://www2.camara.leg.br/legin/fed/lei/2018/lei-13709-14-agosto-2018-787077-publicacaooriginal-156212-pl.html"),
    Law("cdc", "Lei nº 8.078/1990 - Código de Defesa do Consumidor", "CDC",
        "https://www2.camara.leg.br/legin/fed/lei/1990/lei-8078-11-setembro-1990-365086-publicacaooriginal-1-pl.html"),
    Law("mci", "Lei nº 12.965/2014 - Marco Civil da Internet", "Marco Civil",
        "https://www2.camara.leg.br/legin/fed/lei/2014/lei-12965-23-abril-2014-778630-publicacaooriginal-143980-pl.html"),
]
LAW_BY_KEY = {law.key: law for law in LAWS}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")
MAX_CHUNK_CHARS = 900
TOP_K = int(os.getenv("TOP_K", "6"))
