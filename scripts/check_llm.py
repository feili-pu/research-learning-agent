import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv(override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = (os.getenv("RLA_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
    model = os.getenv("RLA_LLM_MODEL", "gpt-4o-mini")
    wire_api = os.getenv("RLA_OPENAI_WIRE_API", "responses").strip().lower()

    print(
        {
            "has_key": bool(api_key),
            "base_url": base_url,
            "model": model,
            "wire_api": wire_api,
        }
    )

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    if wire_api == "chat":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise test assistant."},
                {"role": "user", "content": "用中文回答：接口测试成功了吗？"},
            ],
        )
        print(response.choices[0].message.content)
    else:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "You are a concise test assistant."},
                {"role": "user", "content": "用中文回答：接口测试成功了吗？"},
            ],
        )
        print(response.output_text)


if __name__ == "__main__":
    main()
