"""Unit tests that do not call any external API.

Run with:  python -m pytest tests -q
"""

import os
import subprocess
import sys

import httpx
import openai
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.errors import GENERIC_ERROR_MESSAGE, friendly_error_message


def _response(status):
    return httpx.Response(
        status, request=httpx.Request("POST", "https://api.example.com/v1/chat")
    )


# --- error messages -------------------------------------------------------


def test_rate_limit_error_gets_rate_limit_message():
    exc = openai.RateLimitError("limit", response=_response(429), body=None)
    assert "rate limit" in friendly_error_message(exc).lower()


def test_authentication_error_mentions_api_key():
    exc = openai.AuthenticationError("bad key", response=_response(401), body=None)
    assert ".env" in friendly_error_message(exc)


def test_not_found_error_mentions_model_setting():
    exc = openai.NotFoundError("no model", response=_response(404), body=None)
    assert "LLM_MODEL_NAME" in friendly_error_message(exc)


def test_unknown_error_gets_generic_message():
    assert friendly_error_message(ValueError("boom")) == GENERIC_ERROR_MESSAGE


# --- configuration validation (run in a subprocess to isolate env/imports) --


def _import_settings(env_overrides):
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", "import config.settings"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_unknown_provider_is_rejected():
    result = _import_settings({"LLM_PROVIDER": "anthropic"})
    assert result.returncode != 0
    assert "LLM_PROVIDER must be one of" in result.stderr


def test_missing_openai_key_is_rejected():
    result = _import_settings({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": ""})
    assert result.returncode != 0
    assert "OPENAI_API_KEY is not set" in result.stderr


def test_placeholder_groq_key_is_rejected():
    result = _import_settings(
        {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "gsk_buraya_anahtarini_yaz"}
    )
    assert result.returncode != 0
    assert "GROQ_API_KEY is not set" in result.stderr


# --- ingestion helpers ----------------------------------------------------


def test_find_pdfs_is_case_insensitive_and_ignores_other_files(tmp_path, monkeypatch):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.PDF").write_bytes(b"%PDF")
    (tmp_path / "notes.txt").write_text("not a paper")

    import src.data_ingestion as ingestion

    monkeypatch.setattr(ingestion, "DATA_FILE", str(tmp_path))
    found = [os.path.basename(p) for p in ingestion._find_pdfs()]
    assert found == ["a.pdf", "b.PDF"]


def test_find_pdfs_returns_empty_for_missing_folder(tmp_path, monkeypatch):
    import src.data_ingestion as ingestion

    monkeypatch.setattr(ingestion, "DATA_FILE", str(tmp_path / "missing"))
    assert ingestion._find_pdfs() == []


# --- evaluation helpers ---------------------------------------------------


@pytest.fixture(scope="module")
def run_eval():
    import tests.run_eval as module

    return module


@pytest.mark.parametrize(
    "answer",
    [
        "I don't have that information in the documents.",
        "Bu belgelerde İstanbul'daki hava durumu hakkında bilgi bulunmamaktadır.",
    ],
)
def test_refusal_detection_accepts_both_languages(run_eval, answer):
    assert run_eval.is_refusal(answer)


def test_refusal_detection_rejects_real_answer(run_eval):
    assert not run_eval.is_refusal("ResNet learns residual functions with shortcuts.")


def test_relevance_requires_expected_source_and_keyword(run_eval):
    class FakeNode:
        def __init__(self, file_name, text):
            self.metadata = {"file_name": file_name}
            self._text = text

        def get_content(self):
            return self._text

    right = FakeNode("resnet_1512.03385.pdf", "The residual block adds the input.")
    wrong_paper = FakeNode("yolov1_1506.02640.pdf", "The residual block adds the input.")
    no_keyword = FakeNode("resnet_1512.03385.pdf", "Training details are listed here.")

    assert run_eval.is_relevant(right, "resnet", ["residual"])
    assert not run_eval.is_relevant(wrong_paper, "resnet", ["residual"])
    assert not run_eval.is_relevant(no_keyword, "resnet", ["residual"])


def test_reciprocal_rank(run_eval):
    assert run_eval.reciprocal(1) == 1.0
    assert run_eval.reciprocal(4) == 0.25
    assert run_eval.reciprocal(None) == 0.0
