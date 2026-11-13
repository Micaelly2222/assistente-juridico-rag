from types import SimpleNamespace

import numpy as np
import pytest

from rag import retrieval
from rag.agent import run_agent, run_tool
from rag.answer import LLMAnswer, answer, validate_citations
from rag.ingest import Chunk, clean_text, parse_law, split_long
from rag.config import LAW_BY_KEY
from rag.retrieval import BM25, Retriever, rrf, tokenize

HTML = """<html><body><p>Texto inicial que não é artigo.</p>
<p>CAPÍTULO I</p><p>Art. 1º Esta Lei dispõe sobre dados.</p>
<p>Art. 2º O consumidor pode desistir do contrato no prazo de 7 dias.</p>
<p>Art. 3o O provedor guardará os registros de conexão pelo prazo de 1 ano.</p>
<p>CAPÍTULO II</p><p>Art. 4º Os dados sensíveis exigem consentimento específico.</p>
<p>Brasília, 14 de agosto de 2018.</p></body></html>"""


@pytest.fixture
def chunks():
    return parse_law(LAW_BY_KEY["lgpd"], clean_text(HTML))


@pytest.fixture
def retriever(chunks, monkeypatch):
    # Embeddings falsos e determinísticos: evitam baixar o modelo nos testes
    def fake_embed(texts):
        vocab = ["dados", "desistir", "registros", "sensiveis", "prazo", "consentimento"]
        vecs = np.array([[retrieval.normalize(t).count(w) + 0.01 for w in vocab] for t in texts], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    monkeypatch.setattr(retrieval, "embed", fake_embed)
    vectors = fake_embed([c.text for c in chunks])
    return Retriever(chunks, vectors)


def test_parse_separa_artigos_e_capitulos(chunks):
    assert [c.article for c in chunks] == [1, 2, 3, 4]
    assert chunks[0].heading == "CAPÍTULO I"
    assert chunks[3].heading == "CAPÍTULO II"
    assert "Brasília" not in chunks[3].text


def test_artigo_longo_repete_caput_em_cada_parte():
    text = "Art. 5º Considera-se:\n" + "\n".join(f"{i} - definição número {i} " + "x" * 80 for i in range(20))
    parts = split_long(text, limit=400)
    assert len(parts) > 1
    assert all(p.startswith("Art. 5º Considera-se:") for p in parts)


def test_tokenize_remove_acentos_stopwords_e_plural():
    assert tokenize("Os Dados Pessoais sensíveis") == ["dados", "pesso", "sensivei"]


def test_bm25_prioriza_documento_com_termos_raros():
    bm = BM25(["prazo de garantia", "prazo de arrependimento de sete dias", "outro assunto"])
    assert int(np.argmax(bm.scores("arrependimento"))) == 1


def test_rrf_combina_rankings():
    fused = rrf([np.array([0, 1, 2]), np.array([1, 0, 2])])
    assert fused[2] < fused[0] and fused[2] < fused[1]


def test_busca_hibrida_um_resultado_por_artigo(retriever):
    hits = retriever.search("prazo para desistir", k=3)
    assert hits[0].chunk.article == 2
    assert len({h.chunk.article for h in hits}) == len(hits)


def test_citacoes_fora_do_contexto_sao_removidas(retriever):
    hits = retriever.search("desistir", k=2)
    valid, dropped = validate_citations("Pode desistir [lgpd-art2] e ver [cdc-art99].", [], hits)
    assert "lgpd-art2" in valid
    assert dropped == ["cdc-art99"]


def test_sem_chave_usa_modo_extrativo(retriever, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = answer("prazo para desistir", retriever)
    assert result.mode == "extractive"
    assert result.citations[0] == "lgpd-art2"


class FakeParse:
    def __init__(self, parsed, stop_reason="end_turn"):
        self.parsed, self.stop_reason, self.calls = parsed, stop_reason, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason)


def test_resposta_llm_valida_citacoes_e_contexto_vai_como_dado(retriever):
    fake = FakeParse(LLMAnswer(answer="Sim, em 7 dias [lgpd-art2]. Ver também [mci-art99].",
                               citations=["lgpd-art2"], sufficient=True))
    client = SimpleNamespace(messages=fake)
    result = answer("posso desistir?", retriever, client=client)
    assert result.mode == "llm"
    assert result.citations == ["lgpd-art2"]
    assert result.dropped_citations == ["mci-art99"]
    assert "[mci-art99]" not in result.text
    prompt = fake.calls[0]["messages"][0]["content"]
    assert '<trecho id="lgpd-art2"' in prompt


def test_recusa_volta_para_extrativo(retriever):
    client = SimpleNamespace(messages=FakeParse(None, stop_reason="refusal"))
    assert answer("posso desistir?", retriever, client=client).mode == "extractive"


def block(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


class FakeAgentClient:
    """Simula o modelo: primeiro busca, depois lê o artigo, depois responde."""

    def __init__(self):
        self.turn = 0
        self.messages = self

    def create(self, **kwargs):
        self.turn += 1
        if self.turn == 1:
            content = [block("tool_use", id="t1", name="buscar_leis", input={"consulta": "desistir"})]
            return SimpleNamespace(content=content, stop_reason="tool_use")
        if self.turn == 2:
            content = [block("tool_use", id="t2", name="ler_artigo", input={"lei": "lgpd", "numero": 2})]
            return SimpleNamespace(content=content, stop_reason="tool_use")
        return SimpleNamespace(content=[block("text", text="Prazo de 7 dias [lgpd-art2].")], stop_reason="end_turn")


def test_agente_usa_ferramentas_e_registra_passos(retriever):
    result = run_agent("Posso desistir?", retriever, client=FakeAgentClient())
    assert [s.tool for s in result.steps] == ["buscar_leis", "ler_artigo"]
    assert "7 dias" in result.answer


def test_ferramenta_artigo_inexistente_devolve_erro(retriever):
    assert "não existe" in run_tool(retriever, "ler_artigo", {"lei": "lgpd", "numero": 999})
