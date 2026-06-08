from __future__ import annotations

import os

from openai import OpenAI
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is empty. Fill .env first.")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-5.4"
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Return JSON only: {\"status\": \"ok\"}"}],
        temperature=0,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
