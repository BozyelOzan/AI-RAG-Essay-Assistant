GENERIC_ERROR_MESSAGE = "An error occurred. Please try again."

# Exception class names raised by the Groq and OpenAI clients (both use the
# OpenAI-compatible SDK), mapped to user-facing messages.
_FRIENDLY_MESSAGES = {
    "RateLimitError": (
        "The language model's rate limit was reached. "
        "Please wait a few minutes and try again."
    ),
    "AuthenticationError": (
        "The API key was rejected by the provider. "
        "Check the API key in your .env file."
    ),
    "NotFoundError": (
        "The configured model is not available for this API key. "
        "Check LLM_MODEL_NAME in your .env file."
    ),
}


def friendly_error_message(exc):
    return _FRIENDLY_MESSAGES.get(type(exc).__name__, GENERIC_ERROR_MESSAGE)
