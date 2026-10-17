"""Local, serial document extraction for the isolated Track 2B graph."""

from html.parser import HTMLParser
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict

from .document_store import update_document_status


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.ignored_tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.ignored_tags and self.ignored_tags[-1] == tag:
            self.ignored_tags.pop()

    def handle_data(self, data: str) -> None:
        if not self.ignored_tags:
            self.parts.append(data)


class LocalDocumentIngester:
    def __init__(self, sample_dir: Path) -> None:
        self.sample_dir: Path = sample_dir
        self.lock: Lock = Lock()
        self.state: Dict[str, Any] = {
            "running": False,
            "active_document": None,
            "percent": 0,
            "completed": 0,
            "total": 0,
            "error": None,
        }

    def progress(self) -> Dict[str, Any]:
        with self.lock:
            result: Dict[str, Any] = dict(self.state)

        return result

    def start(self, names: list[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {"ok": False, "error": None}
        selected: list[str] = list(dict.fromkeys(names))
        if not selected:
            result["error"] = "Select at least one document."
        else:
            invalid: list[str] = []
            for name in selected:
                path: Path = self.sample_dir / name
                if Path(name).name != name or path.is_symlink() or not path.is_file():
                    invalid.append(name)

            if invalid:
                result["error"] = "Selected documents are not available in the sample folder."
            else:
                with self.lock:
                    if self.state["running"]:
                        result["error"] = "An ingestion run is already in progress."
                    else:
                        self.state = {
                            "running": True,
                            "active_document": None,
                            "percent": 0,
                            "completed": 0,
                            "total": len(selected),
                            "error": None,
                        }
                        worker: Thread = Thread(target=self._process, args=(selected,), daemon=True)
                        worker.start()
                        result["ok"] = True
                        result["total"] = len(selected)

        return result

    def _process(self, names: list[str]) -> None:
        failure: str | None = None
        completed: int = 0
        for name in names:
            self._set_state(active_document=name, percent=0, completed=completed)
            try:
                processing_result: Dict[str, Any] = update_document_status(name, "Processing")
                if not processing_result["ok"]:
                    raise RuntimeError(self._graph_error(processing_result))

                content: str = self._extract(self.sample_dir / name, name)
                # The platform DocIngester leaves freshly ingested documents at
                # NeedsCategorization; the categorisation skill promotes to Ready
                # only once the categories have actually been reviewed and applied.
                ready_result: Dict[str, Any] = update_document_status(name, "NeedsCategorization", text=content)
                if not ready_result["ok"]:
                    raise RuntimeError(self._graph_error(ready_result))
                completed += 1
            except Exception as error:
                failure = str(error)
                update_document_status(name, "Failed", error=failure[:1000])
                completed += 1
            self._set_state(percent=100, completed=completed, error=failure)

        self._set_state(running=False, active_document=None, percent=100, completed=completed, error=failure)

    def _extract(self, path: Path, name: str) -> str:
        content: str = ""
        suffix: str = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader: Any = PdfReader(str(path))
            pages: int = len(reader.pages)
            parts: list[str] = []
            for index, page in enumerate(reader.pages):
                parts.append(page.extract_text() or "")
                percent: int = int((index + 1) * 100 / max(pages, 1))
                self._set_state(percent=percent)
            content = "\n".join(parts)
        else:
            self._set_state(percent=20)
            content = path.read_text(encoding="utf-8", errors="replace")
            if suffix == ".html":
                parser: _HTMLTextExtractor = _HTMLTextExtractor()
                parser.feed(content)
                content = "\n".join(parser.parts)
            self._set_state(percent=90)

        if not content.strip():
            raise ValueError(f"This PDF format is not supported. No text could be extracted from {name}.")

        return content

    def _set_state(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)

    def _graph_error(self, result: Dict[str, Any]) -> str:
        error: Dict[str, Any] = result.get("error") or {}
        message: str = str(error.get("message") or "Unknown Track 2B Neo4j error")

        return message
