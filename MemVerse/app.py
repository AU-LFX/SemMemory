# app.py
from fastapi import FastAPI, UploadFile, Form

from orchestrator import (
    initialize_rag,
    handle_insert,
    handle_query,
    handle_alfred_start_episode,
    handle_alfred_retrieve_long_term,
    handle_alfred_end_episode,
)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await initialize_rag()
    print("Server started successfully!")
        
@app.post("/insert")
async def insert(query: str = Form(...), video: UploadFile = None, audio: UploadFile = None, image: UploadFile = None):
    try:
        entry = await handle_insert(query, video, audio, image)
        return {"status": "ok", "entry": entry}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/query")
async def query_api(query: str = Form(...), mode: str = Form("hybrid"), use_pm: bool = Form(False)):
    result = await handle_query(query, mode, use_pm)
    return {"status": "ok", **result}

@app.post("/alfred/start_episode")
async def alfred_start_episode(payload: dict):
    return await handle_alfred_start_episode(payload)

@app.post("/alfred/retrieve_long_term")
async def alfred_retrieve_long_term(payload: dict):
    return await handle_alfred_retrieve_long_term(payload)

@app.post("/alfred/end_episode")
async def alfred_end_episode(payload: dict):
    return await handle_alfred_end_episode(payload)
