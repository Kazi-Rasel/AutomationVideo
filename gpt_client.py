# File: /Users/kazirasel/VideoAutomation/System/SysVisuals/LLM/gpt_client.py
# Uses centralized key management via key_loader.py and System/AllKeys.
# Provides a single GPT chat wrapper with quota and sentinel guards.

import sys
from pathlib import Path
from openai import OpenAI, OpenAIError

# Root paths
VA_ROOT = Path.home() / "VideoAutomation"
SYSVIS = VA_ROOT / "System" / "SysVisuals"
ALLKEYS = VA_ROOT / "System" / "AllKeys"
KEYLOADER_DIR = ALLKEYS / "KeyLoader"

# Sentinel file to hard-stop LLM usage
LLM_STOP_FILE = SYSVIS / "LLM" / "LLM_DISABLED"

# ---------------------------------------------------------------------------
# Key loading – ONLY from AllKeys/
# ---------------------------------------------------------------------------

# Prefer `from key_loader import load_key` where key_loader.py is in System/AllKeys/KeyLoader.
# All keys are expected under: VA_ROOT/System/AllKeys/<name>.key
if str(KEYLOADER_DIR) not in sys.path:
    sys.path.insert(0, str(KEYLOADER_DIR))


from key_loader import load_key  # type: ignore[import]


class LLMQuotaError(RuntimeError):
    """Raised when LLM quota or billing issues happen."""
    pass


# ---------------------------------------------------------------------------
# MODEL SELECTION (change here later if you switch to GPT-5.1 or GPT-4o)
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4.1"  # placeholder
# Later you will switch to:
# MODEL_NAME = "gpt-5.1"
# MODEL_NAME = "gpt-5-mini"
# MODEL_NAME = "gpt-4o"
# etc.

# OpenAI client initialized via centralized key loader (or fallback), using "openai.key".
client = OpenAI(api_key=load_key("openai"))


def gpt_chat(
    user_prompt: str,
    system_prompt: str = "",
    temperature: float = 0.3,
) -> str:
    """
    Clean GPT wrapper with:
    - unified model switching
    - quota guard
    - error protection

    Returns:
        String output from LLM.
    """

    # HARD STOP: if sentinel file exists, abort all LLM usage
    if LLM_STOP_FILE.exists():
        raise LLMQuotaError(
            f"LLM disabled: sentinel present at {LLM_STOP_FILE}. "
            f"Fix credit/quota and remove the file."
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    except OpenAIError as e:
        msg = str(e)

        # Detect quota/credit/billing errors
        if (
            "insufficient_quota" in msg.lower()
            or "billing_hard_limit" in msg.lower()
            or "exceeded your current quota" in msg.lower()
            or "credit balance is too low" in msg.lower()
        ):
            # Write STOP file so all future LLM calls stop
            LLM_STOP_FILE.write_text(
                f"LLM AUTO-DISABLED DUE TO QUOTA ERROR:\n{msg}\n",
                encoding="utf-8",
            )
            raise LLMQuotaError("LLM quota exhausted. Pipeline stopped.") from e

        # Any other error → rethrow normally
        raise e
