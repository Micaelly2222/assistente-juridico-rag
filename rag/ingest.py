import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass

import requests
from bs4 import BeautifulSoup

from rag.config import CORPUS, LAWS, MAX_CHUNK_CHARS, RAW, Law

logger = logging.getLogger(__name__)

ARTICLE = re.compile(r"^Art\.\s*(\d+)\s*(?:º|°|o)?\s*[-.–]?\s*(?:-\s*[A-Z]\s*)?", re.MULTILINE)
HEADING = re.compile(r"^(CAP[ÍI]TULO\s+[IVXLC]+|T[ÍI]TULO\s+[IVXLC]+|Se[çc][ãa]o\s+[IVXLC]+)\b.*$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    law: str
    law_short: str
    article: int
    part: int
    heading: str
    text: str


def download(law: Law) -> str:
    path = RAW / f"{law.key}.html"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        response = requests.get(law.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path.read_text(encoding="utf-8", errors="replace")


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = unicodedata.normalize("NFC", soup.get_text("\n"))
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    # Corta o preâmbulo: começa no último título/capítulo antes do Art. 1º (ou no próprio artigo)
    first = ARTICLE.search(text)
    if not first:
        return text
    heading_before = [m.start() for m in HEADING.finditer(text, 0, first.start())]
    return text[heading_before[-1] if heading_before else first.start():]


def split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Quebra artigos longos por incisos/parágrafos, repetindo o caput em cada parte para manter o contexto."""
    if len(text) <= limit:
        return [text]
    lines = text.split("\n")
    caput, rest = lines[0], lines[1:]
    parts, current = [], caput
    for line in rest:
        if len(current) + len(line) + 1 > limit and current != caput:
            parts.append(current)
            current = caput + "\n" + line
        else:
            current += "\n" + line
    parts.append(current)
    return parts


def parse_law(law: Law, text: str) -> list[Chunk]:
    headings = [(m.start(), m.group(0).strip()) for m in HEADING.finditer(text)]
    matches = list(ARTICLE.finditer(text))
    chunks, seen = [], set()
    for i, match in enumerate(matches):
        number = int(match.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.start():end].strip()
        # Fim da lei: assinaturas e anexos depois do último artigo
        body = re.split(r"\n(?:Bras[íi]lia, \d|Este texto não substitui)", body)[0].strip()
        # Artigos citados dentro de outro artigo (alterações em outras leis) não viram chunks próprios
        if number in seen:
            continue
        seen.add(number)
        heading = next((h for pos, h in reversed(headings) if pos < match.start()), "")
        for part, piece in enumerate(split_long(body)):
            chunks.append(Chunk(f"{law.key}-art{number}-{part}", law.key, law.short, number, part, heading, piece))
    return chunks


def build_corpus() -> list[Chunk]:
    chunks = []
    for law in LAWS:
        law_chunks = parse_law(law, clean_text(download(law)))
        articles = len({c.article for c in law_chunks})
        logger.info("%-12s %3d artigos, %4d chunks", law.short, articles, len(law_chunks))
        chunks.extend(law_chunks)
    CORPUS.write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in chunks), encoding="utf-8")
    return chunks


def load_corpus() -> list[Chunk]:
    return [Chunk(**json.loads(line)) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_corpus()
