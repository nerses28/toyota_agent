import sqlite3
from typing import Any, Dict, List

import chromadb
from openai import OpenAI
from smolagents import Tool, ToolCallingAgent, OpenAIServerModel


# --------- utils ---------
def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --------- tools ---------
class SqlSelectTool(Tool):
    name = "sql_select"
    description = "Run read-only SQL SELECT queries against the local SQLite database."
    inputs = {
        "query": {
            "type": "string",
            "description": "SQL SELECT statement to execute. Only SELECT is allowed.",
            "nullable": True,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of rows to return.",
            "default": 100,
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path

    def forward(self, query: str | None = None, limit: int | None = 100) -> str:
        if not query or not str(query).strip():
            return "Error: 'query' is required."
        q = str(query).strip().rstrip(";")
        if not q.lower().startswith("select"):
            return "Error: only SELECT queries are allowed."
        lim = int(limit) if (isinstance(limit, int) or str(limit).isdigit()) else 100

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(q)
            rows = cur.fetchmany(lim)
            cols = [d[0] for d in cur.description] if cur.description else []
        finally:
            conn.close()

        # Return as simple CSV text
        import io, csv
        buf = io.StringIO()
        w = csv.writer(buf)
        if cols:
            w.writerow(cols)
        w.writerows(rows)
        return buf.getvalue().strip() or "(no rows)"


class ChromaRagTool(Tool):
    name = "rag_search"
    description = (
        "Retrieve relevant passages from the local Chroma DB built from PDFs. "
        "Returns top-k chunks with basic citations."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Natural language question.",
            "nullable": True,
        },
        "k": {
            "type": "integer",
            "description": "Number of passages to retrieve.",
            "default": 5,
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedding_model: str = "text-embedding-3-small",
    ):
        super().__init__()
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(collection_name)
        self.oai = OpenAI()
        self.embedding_model = embedding_model

    def _embed(self, text: str) -> List[float]:
        resp = self.oai.embeddings.create(model=self.embedding_model, input=text)
        return resp.data[0].embedding

    def forward(self, query: str | None = None, k: int | None = 5) -> str:
        if not query or not str(query).strip():
            return "Error: 'query' is required."
        topk = int(k) if (isinstance(k, int) or str(k).isdigit()) else 5

        qvec = self._embed(str(query))
        res: Dict[str, Any] = self.collection.query(
            query_embeddings=[qvec],
            n_results=topk,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        if not docs:
            return "No relevant passages found."

        parts = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
            meta = meta or {}
            src = meta.get("file_path") or meta.get("source") or "unknown"
            page = meta.get("page") or meta.get("page_number")
            parts.append(f"[{i}] score={dist:.4f} source={src} page={page}\n{doc.strip()}")
        return "\n\n".join(parts)


# --------- agent wiring ---------
def build_agent(
    system_prompt_path: str = "system_prompt.txt",
    db_path: str = "toyota.db",
    chroma_dir: str = "chroma",
    collection_name: str = "docs",
) -> ToolCallingAgent:
    system_prompt = load_system_prompt(system_prompt_path)

    sql_tool = SqlSelectTool(db_path=db_path)
    rag_tool = ChromaRagTool(persist_dir=chroma_dir, collection_name=collection_name)

    model = OpenAIServerModel(model_id="gpt-4o")

    agent = ToolCallingAgent(
        tools=[sql_tool, rag_tool],
        model=model,
        instructions=system_prompt,
        verbosity_level=2,
        max_steps=8,
    )
    return agent


if __name__ == "__main__":
    AGENT = build_agent(
        system_prompt_path="system_prompt.txt",
        db_path="toyota.db",
        chroma_dir="chroma",
        collection_name="docs",
    )

    '''
    user_task = "What is the standard Toyota warranty for Europe?"
    while True:
        print('=' * 20)
        user_task = input('Ask your question: ')
        result = AGENT.run(user_task)
        print("\n" + str(result))
    '''
    '''
    questions = [
        "Compare Toyota vs Lexus SUV sales in Western Europe in 2024 and summarize any key warranty differences.",
        "I drive rideshare in Germany with a 2023 Yaris Hybrid. Are there any special warranty limits for commercial/taxi use, and how many Yaris Hybrid Fleet/Taxi sales were recorded in Germany in 2023?",
        "I’m choosing between a GT86 and a Hilux 48V. What are the recommended fuel/energy requirements or cautions for each, and how did UK sales for these two models compare in 2022 vs 2023?",
        "My 2024 Hilux 48V just showed a hybrid/48V warning. What does the manual say I should do immediately, and how many Hilux units were sold in France in Q1 vs Q2 of 2024?",
        "I’m buying a Lexus RC in Italy but may move to France next year. Will the warranty be honored by authorized dealers across the EU, and which three EU countries had the highest Lexus RC sales in 2023?"
    ]
    '''
    questions = ["What is the standard Toyota warranty for Europe?",
                 "Monthly RAV4 HEV sales in Germany in 2024."
    ]

    results = []
    for user_task in questions:
        print('=' * 20)
        result = AGENT.run(user_task)
        print("\n" + str(result))
        results.append(str(result))

    print('*' * 20)
    for result in results:
        print(result)
        print('-------------' * 20)
