import os
from typing import List, Iterable, Tuple, Dict, Any, Union
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, Embeddings
from openai import OpenAI
from PyPDF2 import PdfReader

class OpenAIEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Chroma-compatible embedding function using OpenAI Embeddings API."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        resp = self.client.embeddings.create(model=self.model, input=input)
        return [d.embedding for d in resp.data]


class BuildEmbeddings:
    def __init__(
        self,
        source_dir: Union[str, List[str]],
        collection_name: str,
        persist_dir: str = "./chroma",
        recursive: bool = True,
        allowed_ext: set[str] = {".pdf"},
        batch_size: int = 64,
        embed_model: str = "text-embedding-3-small",
        openai_api_key: str | None = None,
        distance: str = "cosine",
    ):
        self.source_dirs = [source_dir] if isinstance(source_dir, str) else list(source_dir)
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.recursive = recursive
        self.allowed_ext = allowed_ext
        self.batch_size = batch_size

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._ef = OpenAIEmbeddingFunction(model=embed_model, api_key=openai_api_key)
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": distance},
        )

    def _iter_files(self) -> Iterable[str]:
        for root_dir in self.source_dirs:
            if not os.path.isdir(root_dir):
                continue
            if self.recursive:
                for r, _, files in os.walk(root_dir):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in self.allowed_ext:
                            yield os.path.join(r, f)
            else:
                for f in os.listdir(root_dir):
                    full = os.path.join(root_dir, f)
                    if os.path.isfile(full) and os.path.splitext(f)[1].lower() in self.allowed_ext:
                        yield full

    def _count_total_pages(self, files: List[str]) -> int:
        total = 0
        for fp in files:
            try:
                reader = PdfReader(fp)
                total += len(reader.pages)
            except Exception:
                continue
        return total

    def _page_key(self, file_path: str, page_num: int) -> str:
        rel = os.path.relpath(file_path)
        return f"{rel}::page:{page_num}"

    def _page_metadata(self, file_path: str, page_num: int) -> Dict[str, Any]:
        rel = os.path.relpath(file_path)
        base = os.path.basename(file_path)
        return {
            "source": rel,
            "file": base,
            "page": page_num,
            "file_page": f"{base}_{page_num}",
            "uri": f"{rel}#page={page_num}",
        }

    def _iter_pages(self, file_path: str) -> Iterable[Tuple[int, str]]:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = text.strip()
            if text:
                yield i, text

    def build(self) -> int:
        files = list(self._iter_files())
        total_pages = self._count_total_pages(files)
        if total_pages == 0:
            print("No pages to index.")
            return 0

        done = 0
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for fp in files:
            for page_num, text in self._iter_pages(fp):
                docs.append(text)
                metas.append(self._page_metadata(fp, page_num))
                ids.append(self._page_key(fp, page_num))
                done += 1

                if done % self.batch_size == 0 or done == total_pages:
                    pct = (done / total_pages) * 100
                    print(f"[{done}/{total_pages}] {pct:.1f}% indexed", flush=True)

                # Flush by batch
                if len(docs) >= self.batch_size:
                    self._flush_batch(ids, docs, metas)
                    ids, docs, metas = [], [], []

        # Flush tail
        if docs:
            self._flush_batch(ids, docs, metas)

        return done

    def _flush_batch(self, ids: List[str], docs: List[str], metas: List[Dict[str, Any]]) -> None:
        try:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
        except AttributeError:
            try:
                self.collection.delete(ids=ids)
            except Exception:
                pass
            self.collection.add(ids=ids, documents=docs, metadatas=metas)


def main():
    builder = BuildEmbeddings(
        source_dir=["docs", "manuals"],
        collection_name="docs",
        persist_dir="./chroma",
        recursive=True,
        allowed_ext={".pdf"},
        batch_size=64,
        embed_model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        distance="cosine",
    )
    total = builder.build()
    print(f"Indexed pages: {total}")


if __name__ == "__main__":
    main()
