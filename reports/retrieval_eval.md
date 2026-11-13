# Retrieval evaluation

30 questions written in plain language, each with the article(s) that answer it.
A hit means the right article appears in the top k results (one result per article).

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR@10 |
|---|---|---|---|---|
| bm25 | 0.53 | 0.80 | 0.83 | 0.66 |
| dense | 0.63 | 0.87 | 0.87 | 0.75 |
| hybrid | 0.70 | 0.80 | 0.90 | 0.78 |

Questions outside the top 5 in `hybrid` mode:

- Quando o tratamento de dados pessoais deve terminar?
- Quando o juiz pode inverter o ônus da prova a favor do consumidor?
- Quais direitos tem o usuário de internet no Brasil?
