"""Quick smoke test for loading the OpenAI API key and making one request."""

import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

dotenv_path = find_dotenv(usecwd=True)
print("dotenv path:", dotenv_path)

load_dotenv(dotenv_path, override=True)

key = os.getenv("OPENAI_API_KEY")

if not key:
    raise ValueError("OPENAI_API_KEY was not loaded. Check your .env file.")

print("Using key ending:", key[-4:])

client = OpenAI(api_key=key)

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Say test only."
)

print(response.output_text)
