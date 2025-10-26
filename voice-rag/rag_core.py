# rag_core.py — 100% COMPATIBLE con openai==2.6.1
from pathlib import Path
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Any, Dict, Optional

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPEN_AI_API_KEY)

# --------------------------- Vector Store (cliente principal) ---------------------------

def _find_or_create_vector_store(store_name: str):
    # Listar todos los vector stores
    try:
        stores = client.vector_stores.list(limit=100)
        for vs in stores.data:
            if vs.name == store_name:
                return vs
    except Exception as e:
        print(f"Error listando vector stores: {e}")

    # Crear si no existe
    return client.vector_stores.create(name=store_name)

def _wait_file_indexed(vector_store_id: str, file_id: str, timeout_s=180):
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            file = client.vector_stores.files.retrieve(
                vector_store_id=vector_store_id,
                file_id=file_id
            )
            if file.status == "completed":
                return
            if file.status in ("failed", "cancelled"):
                raise RuntimeError(f"Indexing failed: {file.status}")
        except:
            pass
        time.sleep(2)
    raise TimeoutError("File indexing timeout")

def upload_file_to_vector_store(path: str, store_name: str = "Some Fake Football Articles", wait_index: bool = True):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    vs = _find_or_create_vector_store(store_name)

    # Ver si el archivo ya existe por nombre
    try:
        files = client.vector_stores.files.list(vector_store_id=vs.id)
        for f in files.data:
            try:
                meta = client.files.retrieve(f.id)
                if meta.filename == p.name:
                    if wait_index:
                        _wait_file_indexed(vs.id, f.id)
                    return vs.id, f.id
            except:
                continue
    except Exception as e:
        print(f"Error verificando archivos: {e}")

    # Subir archivo
    with open(p, "rb") as f:
        file_obj = client.files.create(file=f, purpose="assistants")

    # Adjuntar al vector store
    client.vector_stores.files.create(
        vector_store_id=vs.id,
        file_id=file_obj.id
    )

    if wait_index:
        _wait_file_indexed(vs.id, file_obj.id)

    return vs.id, file_obj.id

# --------------------------- Assistant (beta) ---------------------------

_ASSISTANT_ID = None

def _get_or_create_assistant(vector_store_id: str):
    global _ASSISTANT_ID
    if _ASSISTANT_ID:
        try:
            return client.beta.assistants.retrieve(_ASSISTANT_ID)
        except:
            pass

    # Buscar por metadata
    try:
        assistants = client.beta.assistants.list(limit=100)
        for a in assistants.data:
            if a.metadata.get("app") == "parmafc-rag":
                _ASSISTANT_ID = a.id
                return a
    except:
        pass

    # Crear asistente con file_search + vector store
    assistant = client.beta.assistants.create(
        name="ParmaFC RAG Assistant",
        instructions=(
            "You are a helpful football analyst. Use only the provided articles. "
            "Answer using these sections: Strengths, Weaknesses, EvolutionOverTime, Sources. "
            "Be concise and neutral."
        ),
        model=OPEN_AI_MODEL,
        tools=[{"type": "file_search"}],
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store_id]
            }
        },
        metadata={"app": "parmafc-rag"}
    )
    _ASSISTANT_ID = assistant.id
    return assistant

# --------------------------- RAG ---------------------------

def generate_response(
    prompt: str,
    vector_store_id: str,
    temperature: float = 0.0,
    max_output_tokens: Optional[int] = None,
    structured_sections: bool = True,
) -> Dict[str, Any]:

    assistant = _get_or_create_assistant(vector_store_id)

    # Crear thread
    thread = client.beta.threads.create()

    # Enviar mensaje
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )

    # Ejecutar run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        temperature=temperature,
        max_completion_tokens=max_output_tokens,
    )

    # Esperar completado
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status in ["completed", "failed", "cancelled"]:
            break
        time.sleep(1)

    if run.status != "completed":
        return {"status": "error", "text": f"Run failed: {run.status}"}

    # Obtener respuesta
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    if not messages.data:
        return {"status": "error", "text": "No response"}

    text = messages.data[0].content[0].text.value

    # Parseo de secciones
    sections = None
    if structured_sections and text:
        def _grab(header, txt):
            if header in txt:
                part = txt.split(header, 1)[1]
                for h in ["Strengths:", "Weaknesses:", "EvolutionOverTime:", "Sources:"]:
                    if h != header and h in part:
                        part = part.split(h, 1)[0]
                return part.strip()
            return None

        sections = {
            "Strengths": _grab("Strengths:", text),
            "Weaknesses": _grab("Weaknesses:", text),
            "EvolutionOverTime": _grab("EvolutionOverTime:", text),
            "Sources": _grab("Sources:", text),
        }

    return {
        "status": "ok",
        "text": text.strip(),
        "sections": sections
    }

# --------------------------- Wrappers ---------------------------

def ensure_vector_store(store_name: str, path: str):
    vs_id, _ = upload_file_to_vector_store(path, store_name, wait_index=True)
    return vs_id

def rag_answer(query: str, vector_store_id: str, temperature: float = 0.0, structured_sections: bool = True):
    result = generate_response(
        prompt=query,
        vector_store_id=vector_store_id,
        temperature=temperature,
        structured_sections=structured_sections
    )
    return result.get("text", "").strip()