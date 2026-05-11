"""
langchain_chain.py
------------------
LangChain chain that sends a base64-encoded image to GPT-4o-mini
and returns a natural-language description of the image.
"""

import base64
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def _build_llm() -> ChatOpenAI:
    """Instantiate the GPT-4o-mini chat model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.\n"
            "Add it to your .env file: OPENAI_API_KEY=sk-xxxx"
        )
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        max_tokens=1024,
    )


def describe_image(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    """
    Send image_bytes to GPT-4o-mini and return a text description.
    Used by mai.py (FastAPI endpoint) and image.py (graph analysis).
    """
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Please describe this image in detail. "
                    "Include objects, colors, scene context, and any notable features."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{b64_image}",
                    "detail": "auto",
                },
            },
        ]
    )

    llm    = _build_llm()
    parser = StrOutputParser()
    chain  = llm | parser
    return chain.invoke([message])


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python langchain_chain.py <image_path>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        raw = f.read()

    ext      = sys.argv[1].rsplit(".", 1)[-1].lower()
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
    mime     = mime_map.get(ext, "image/jpeg")

    result = describe_image(raw, mime)
    print("\n=== Image Description ===")
    print(result)
