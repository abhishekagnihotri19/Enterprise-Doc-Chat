import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from logger import global_logger as log
from src.data_ingestion import ChatIngestor
from utils.doc_ops import FastApiFileHandler
from src.retrieval import ConversationalRag


app= FastAPI(title="Document Chatting System", version="0.1")



FAISS_BASE= os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE= os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME= os.getenv("FAISS_INDEX_NAME", "index")

BASE_DIR= Path(__file__).resolve().parent.parent

app.mount("/static", StaticFiles(directory=str(BASE_DIR/"static")), name= "static")

templates = Jinja2Templates(directory=str(BASE_DIR/"templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

@app.get("/", response_class= HTMLResponse)
async def serve_ui(request:Request):
    log.info ("Serving UI Homepage")
    resp = templates.TemplateResponse("index.html",{"request":request})
    resp.headers["cache-control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str,str]:
    log.info("Health checked passed")
    return {"status":"ok", "service":"Document_CHatting"}

#--------------------CHAT INDEX--------------------#

@app.post("/chat/index")
async def chat_build_index (
    files:List[UploadFile]=File(...),
    session_id:Optional[str]= Form(None),
    use_session_dirs:bool = True,
    chunk_size:int= Form(1000),
    chunk_overlap:int =Form(200),
    k:int= Form (5)
) ->Any:
    try:
      log.info(f"Indexing chat Session", Session_id= {session_id}, Files=[f.filename for f in files]) #Extracting file name from group of files
      wrapped= [FastApiFileHandler(f) for f in files] # Through FastAPiFileHandler: Convert FAST API File object into Python readable file name

      #Remember : .filename is file name get through FASTAPT::::But, file.name is use in Python for Reading Purpose
    
        # Calling CLass ChatIngestor present in DataIngestion through Object(ci)
      ci= ChatIngestor(
          temp_base = UPLOAD_BASE,
          faiss_base = FAISS_BASE,
          use_session_dirs = use_session_dirs,
          session_id = session_id or None,
    
      )

      # Call MethoD to Built Retriver

      ci.built_in_retrieval (
          wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

      log.info (f"Index creating Sucessfully for session {ci.session_id}")
      return ({"session_id": ci.session_id, "k":k, "use_session_dirs": use_session_dirs})

    except HTTPException:
        raise
    except Exception as e:
        log.exception("chat index building failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

#----------------CHAT QUERY----------------------#
@app.post("/chat/query")
async def chat_query (
    question:str = Form(...),
    session_id: Optional[str]= Form(None),
    use_session_dir:bool= Form(True),
    k:int=Form (5),
)-> Any:
     try:
        log.info ("Received Chat Query '{question}' | session: {session_id}")
        if use_session_dir and not session_id:
            raise HTTPException(status_code=400, detail= "Session_id is Required when use_session_directory is True")
        
        index_dir= os.path.join(FAISS_BASE, session_id) if use_session_dir else FAISS_BASE
        if not os.path.isdir(index_dir):
            raise HTTPException (status_code= 404, details=f"FAISS Index is not found at {index_dir}")
        
        rag= ConversationalRag (session_id=session_id)
        rag.load_retriever_from_faiss (index_dir, k=5, index_name= FAISS_INDEX_NAME)
        response=rag.invoke(question, chat_history=[])
        log.info ("Chat Query Handled Succesfully")

        return{
            "answer": response,
            "session_id":session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }

     except HTTPException:
         raise
     except Exception as e:
         log.exception ("chat query failed")
         raise HTTPException(status_code=500, detail=f"Query failed: {e}")
         






#uvicorn api.main:app --port 8080 --reload 