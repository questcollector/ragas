from __future__ import annotations

import typing as t
import json
import os
import logging

P_enc = t.ParamSpec("P_enc")
P_dec = t.ParamSpec("P_dec")
R = t.TypeVar("R", bound=t.List[int], covariant=True)

logger = logging.getLogger(__name__)


class TokenizerProtocol(t.Protocol, t.Generic[P_enc, P_dec, R]):
    def encode(self, *args: P_enc.args, **kwargs: P_enc.kwargs) -> R:
        """Encode a string into a list of token IDs."""
        ...

    def decode(self, *args: P_dec.args, **kwargs: P_dec.kwargs) -> str:
        """Decode a list of token IDs back into a string."""
        ...


def _safe_json_loads_(s: str | None) -> dict:
    """
    Safely load a JSON string into a dictionary.
    If the string is empty or None, return an empty dictionary.
    """
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON string {s!r}") from e


_HF_TOKENIZER: TokenizerProtocol


def huggingface_tokenizer() -> TokenizerProtocol | None:
    """
    Returns the Huggingface tokenizer if available, otherwise raises an error.
    The tokenizer is loaded from the environment variable `HF_TOKENIZER_MODEL`.
    If the environment variable is not set, it raises a NameError.
    """
    try:
        global _HF_TOKENIZER
        if "_HF_TOKENIZER" in globals():
            return _HF_TOKENIZER

        # Attempt to load Huggingface tokenizer if available
        from transformers import AutoTokenizer

        hf_tokenizer_model = os.environ.get("HF_TOKENIZER_MODEL")
        if not hf_tokenizer_model:
            raise NameError(
                "HF_TOKENIZER_MODEL environment variable must be set to use Huggingface tokenizer."
            )
        hf_tokenizer_args = os.environ.get("HF_TOKENIZER_ARGS")
        tokenizer_args = _safe_json_loads_(hf_tokenizer_args)

        os.environ["TOKENIZERS_PARALLELISM"] = (
            "false"  # Disable parallelism for tokenizers
        )

        _HF_TOKENIZER = AutoTokenizer.from_pretrained(
            hf_tokenizer_model, **tokenizer_args
        )
        logger.warning(f"Huggingface Tokenizer has been loaded: {hf_tokenizer_model}")
        return _HF_TOKENIZER
    except Exception as e:
        if not isinstance(e, NameError):
            # Only warn if it's not a NameError (which is expected if the env var is not set)
            # This avoids warning when the user is not using Huggingface tokenizer
            raise RuntimeError(
                "Failed to load Huggingface tokenizer. Unset environment variable 'HF_TOKENIZER_MODEL' to disable using huggingface tokenizer."
            ) from e
        return None
