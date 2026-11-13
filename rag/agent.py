import json
import logging
from dataclasses import dataclass, field

import anthropic

from rag.answer import RULES as ANSWER_RULES
from rag.config import LAW_BY_KEY, LLM_MODEL
from rag.retrieval import Retriever

logger = logging.getLogger(__name__)

MAX_STEPS = 6

TOOLS = [
    {
        "name": "buscar_leis",
        "description": "Busca trechos de lei relevantes para uma consulta. Use uma consulta por assunto. "
                       "Pode filtrar por lei: lgpd, cdc ou mci.",
        "input_schema": {
            "type": "object",
            "properties": {
                "consulta": {"type": "string", "description": "O que procurar, em português"},
                "lei": {"type": "string", "enum": ["lgpd", "cdc", "mci"], "description": "Opcional"},
            },
            "required": ["consulta"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "ler_artigo",
        "description": "Lê o texto completo de um artigo específico.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lei": {"type": "string", "enum": ["lgpd", "cdc", "mci"]},
                "numero": {"type": "integer", "minimum": 1},
            },
            "required": ["lei", "numero"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

AGENT_SYSTEM = (
    "Você é um assistente de pesquisa jurídica. Para responder, use as ferramentas: busque os artigos, "
    "leia os que forem relevantes e só então responda. Perguntas com mais de um assunto podem exigir buscas "
    "em leis diferentes.\n\n" + ANSWER_RULES
)


@dataclass
class Step:
    tool: str
    input: dict
    output_preview: str


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_reason: str = "end_turn"


def run_tool(retriever: Retriever, name: str, args: dict) -> str:
    if name == "buscar_leis":
        hits = retriever.search(args["consulta"], k=5, law=args.get("lei"))
        return json.dumps(
            [{"id": f"{h.chunk.law}-art{h.chunk.article}", "lei": h.chunk.law_short,
              "trecho": h.chunk.text[:400]} for h in hits],
            ensure_ascii=False,
        )
    if name == "ler_artigo":
        law, number = args["lei"], int(args["numero"])
        chunks = retriever.article(law, number)
        if not chunks:
            return json.dumps({"erro": f"artigo {number} não existe na {LAW_BY_KEY[law].short}"}, ensure_ascii=False)
        text = "\n".join(c.text if i == 0 else "\n".join(c.text.split("\n")[1:]) for i, c in enumerate(chunks))
        return json.dumps({"id": f"{law}-art{number}", "texto": text}, ensure_ascii=False)
    return json.dumps({"erro": f"ferramenta desconhecida: {name}"})


def run_agent(question: str, retriever: Retriever, client: anthropic.Anthropic | None = None) -> AgentResult:
    client = client or anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]
    result = AgentResult(answer="")

    for _ in range(MAX_STEPS):
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=4096,
            system=AGENT_SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "low"},
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            result.answer = "Não foi possível responder a esta pergunta."
            result.stopped_reason = "refusal"
            return result

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            result.answer = "\n".join(b.text for b in response.content if b.type == "text").strip()
            result.stopped_reason = response.stop_reason
            return result

        tool_results = []
        for block in tool_uses:
            output = run_tool(retriever, block.name, block.input)
            result.steps.append(Step(block.name, dict(block.input), output[:300]))
            logger.info("ferramenta %s %s", block.name, block.input)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": tool_results})

    result.answer = "Limite de passos atingido antes de uma resposta final."
    result.stopped_reason = "max_steps"
    return result
