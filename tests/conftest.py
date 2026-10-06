import pytest

from src.models import llm_client


@pytest.fixture(autouse=True)
def isolate_llm_service(monkeypatch):
    """Unit tests must not depend on a resident model server.

    Tests exercising model responses can override this stub locally.
    """
    def unavailable(*args, **kwargs):
        raise RuntimeError("LLM service is disabled in isolated unit tests")

    monkeypatch.setattr(llm_client, "_openai_chat_completion", unavailable)
