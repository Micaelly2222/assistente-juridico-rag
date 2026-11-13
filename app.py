import json
import os

import pandas as pd
import streamlit as st

# No Streamlit Cloud a chave vem dos secrets; localmente, do .env
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except FileNotFoundError:
    pass

from rag.agent import run_agent  # noqa: E402
from rag.answer import answer, llm_available  # noqa: E402
from rag.config import LAW_BY_KEY, LAWS, REPORTS  # noqa: E402
from rag.retrieval import Retriever  # noqa: E402

st.set_page_config(page_title="Assistente Jurídico RAG", layout="wide")

EXAMPLES = [
    "Comprei pela internet e me arrependi. Em quantos dias posso desistir?",
    "Quais direitos eu tenho sobre os meus dados pessoais?",
    "A plataforma responde por um post ofensivo de um usuário?",
    "Por quanto tempo um aplicativo precisa guardar os registros de acesso?",
    "A empresa precisa avisar quando houver vazamento de dados?",
    "Fui cobrado indevidamente. Recebo o valor em dobro?",
]


@st.cache_resource(show_spinner="Carregando índice e modelo de embeddings...")
def retriever() -> Retriever:
    r = Retriever()
    r.search("aquecimento", k=1)
    return r


st.title("Assistente Jurídico com RAG")
st.caption(
    "Perguntas sobre LGPD, Código de Defesa do Consumidor e Marco Civil da Internet, respondidas com o artigo "
    "da lei como fonte. Busca híbrida (BM25 + embeddings), respostas com citação verificada e agente com "
    "ferramentas. [Código no GitHub](https://github.com/Micaelly2222/assistente-juridico-rag)"
)

with st.sidebar:
    st.header("Configuração")
    law_filter = st.selectbox("Lei", ["Todas"] + [law.short for law in LAWS])
    law_key = next((law.key for law in LAWS if law.short == law_filter), None)
    st.caption(f"LLM: {'ligado (Claude)' if llm_available() else 'desligado, modo extrativo'}")
    st.caption("Não é aconselhamento jurídico. Texto das leis: publicação original no portal da Câmara.")

tab_ask, tab_agent, tab_search, tab_eval = st.tabs(["Perguntar", "Agente", "Busca", "Avaliação"])

with tab_ask:
    example = st.selectbox("Exemplos", EXAMPLES, index=None, placeholder="Escolha uma pergunta de exemplo", key="ex_ask")
    question = st.text_input("Sua pergunta", value=example or "", key="q_ask")
    if st.button("Responder", type="primary") and question:
        with st.spinner("Buscando os artigos..."):
            result = answer(question, retriever(), law=law_key)
        if result.mode == "extractive":
            st.info("Modo extrativo: sem chave de API, o sistema mostra os artigos mais relevantes em vez de gerar texto.")
        elif not result.sufficient:
            st.warning("Os artigos encontrados podem não responder totalmente à pergunta.")
        st.markdown(result.text)
        if result.dropped_citations:
            st.caption(f"Citações removidas por não estarem nas fontes: {', '.join(result.dropped_citations)}")
        st.subheader("Fontes")
        for h in result.sources:
            cid = f"{h.chunk.law}-art{h.chunk.article}"
            mark = " ✓ citado" if cid in result.citations else ""
            with st.expander(f"{LAW_BY_KEY[h.chunk.law].short}, art. {h.chunk.article} ({h.chunk.heading}){mark}"):
                st.text(h.chunk.text)

with tab_agent:
    st.write(
        "O agente decide quais buscas fazer e quais artigos ler antes de responder. "
        "Bom para perguntas que cruzam mais de uma lei."
    )
    if not llm_available():
        st.info("O agente precisa de uma chave da API da Anthropic (ANTHROPIC_API_KEY nos secrets).")
    else:
        q = st.text_input(
            "Pergunta",
            value="Uma loja online vazou meus dados e ainda me cobrou em duplicidade. Quais direitos eu tenho?",
            key="q_agent",
        )
        if st.button("Perguntar ao agente", type="primary"):
            with st.spinner("O agente está pesquisando..."):
                result = run_agent(q, retriever())
            st.markdown(result.answer)
            st.subheader("Passos do agente")
            st.dataframe(
                pd.DataFrame([{"Ferramenta": s.tool, "Entrada": json.dumps(s.input, ensure_ascii=False)}
                              for s in result.steps]),
                hide_index=True, width="stretch",
            )

with tab_search:
    q = st.text_input("Consulta", value="prazo de guarda de registros", key="q_search")
    mode = st.radio("Modo", ["hybrid", "bm25", "dense"], horizontal=True)
    if q:
        hits = retriever().search(q, k=8, mode=mode, law=law_key)
        st.dataframe(
            pd.DataFrame([{
                "Lei": h.chunk.law_short, "Artigo": h.chunk.article, "Score final": round(h.score, 4),
                "BM25": round(h.bm25, 2), "Similaridade": round(h.dense, 3), "Trecho": h.chunk.text[:160],
            } for h in hits]),
            hide_index=True, width="stretch",
        )

with tab_eval:
    st.write(
        "30 perguntas em linguagem comum, cada uma com o artigo que a responde. "
        "Recall@k: o artigo certo aparece entre os k primeiros resultados."
    )
    path = REPORTS / "retrieval_eval.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        st.dataframe(
            pd.DataFrame([{k: v for k, v in r.items() if k != "misses_top5"} for r in rows]),
            hide_index=True, width="stretch",
        )
        st.caption("A busca híbrida combina BM25 e embeddings com Reciprocal Rank Fusion.")
