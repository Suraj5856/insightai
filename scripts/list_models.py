import httpx

from app.config import settings

if not settings.google_api_key:
    raise SystemExit("GOOGLE_API_KEY is empty. Add it to .env first.")

r = httpx.get(
    "https://generativelanguage.googleapis.com/v1beta/models",
    params={"key": settings.google_api_key, "pageSize": 200},
    timeout=30,
)
if r.status_code != 200:
    raise SystemExit(f"Error {r.status_code}: {r.text[:300]}")

chat, embed = [], []
for m in r.json().get("models", []):
    name = m["name"].replace("models/", "")
    methods = m.get("supportedGenerationMethods", [])
    if "embedContent" in methods:
        embed.append(name)
    elif "generateContent" in methods and "gemini" in name:
        chat.append(name)

print("\nCHAT models (use for GEMINI_LLM_MODEL):")
print("\n".join(f"  {n}" for n in sorted(chat)))
print("\nEMBEDDING models (use for GEMINI_EMBEDDING_MODEL):")
print("\n".join(f"  {n}" for n in sorted(embed)))