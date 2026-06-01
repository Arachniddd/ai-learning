from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from llm_client import *
from splitter import split_by_paragraph
from rag_store import *
from vector_store import *
import time
from logger import *
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/")
def root():
    return {"message": "helloapi"}

@app.post("/split")
def split_text(request: SplitRequest):
    return {"chunks": split_by_paragraph(request.text)}

@app.post("/summarize")
def summarize_text(request : SummerizeRequest):
    result = summarize_note(request.text)
    return result

# @app.post("/adddocument")
# def add_doc(request : AdddocumentRequest):
#     doc_chunks = add_document(request.text, request.source)

#     return {
#         "messsage" : "Document added",
#         "chunk_count" : len(doc_chunks),
#         "chunks" : doc_chunks
#     }

# @app.post("/ask")
# def ask(request : AskRequest):
#     contexts = search_chunks(request.question, request.top_k)
    
#     if len(contexts) == 0:
#         return {
#             "message" : "Nothing can be searched!\nPlease try another question or add more documents.",
#             "context" : []
#         }
    
#     result = answer_with_context(request.question, contexts)

#     return {
#         "answer" : result["answer"],
#         "used_chunks" : result.get("used_chunks", []),
#         "contexts" : contexts
#     }


@app.post("/adddocument")
def add_docu(request : AdddocumentRequest):
    doc_chunks = add_document_to_vector_store(request.text, request.source)

    return {
        "messsage" : "Document added",
        "chunk_count" : len(doc_chunks),
        "chunks" : doc_chunks
    }

@app.post("/ask")
def ask(request : AskRequest):
    contexts = search_vector_store(request.question, request.top_k)
    
    if len(contexts) == 0:
        return {
            "message" : "Nothing can be searched!\nPlease try another question or add more documents.",
            "context" : []
        }
    
    result = answer_with_context(request.question, contexts)

    return {
        "answer" : result["answer"],
        "used_chunks" : result.get("used_chunks", []),
        "contexts" : contexts
    }

@app.post("/upload_file")
async def upload_file(file : UploadFile = File(...)):
    filename = file.filename

    if not filename.endswith((".txt", ".md")):
        return {
            "error": "Only .txt and .md file are supported now."
        }
    
    content_bytes = await file.read()

    text = content_bytes.decode("utf-8")

    added_chunks = add_document_to_vector_store(
        text=text,
        source=filename
    )

    return{
        "message": "file uploaded and added to vector store",
        "filename": filename,
        "chunk_count": len(added_chunks),
        "chunks": added_chunks
    }

@app.get("/chunks")
def get_chunks(limit : int = 20):
    chunks = list_all_chunks(limit)

    return {
        "count" : len(chunks),
        "chunks" : chunks
    }

@app.delete("/chunks")
def delete_chunks():
    num = clear_vector_store()
    return {
        "deleted_count" : num
    }

@app.post("/agent")
def agent(request : AgentRequest):
    start_time = time.time()

    trace = {
        "message": request.message,
        "decision": None,
        "tool_name": None,
        "tool_arguments": None,
        "tool_result": None,
        "answer": None,
        "error": None
    }
    try:
        decision = decide_tool_use(message=request.message)
        trace["decision"] = decision

        if not decision.get("need_tool"):
            answer = decision.get("answer","")

            trace["answer"] = answer
            trace["duration_ms"] = int((time.time() - start_time) * 1000)

            write_agent_log(trace)

            return {
                "mode": "direct_answer",
                "answer": decision.get("answer", ""),
                "trace": trace
            }
        
        tool_name = decision.get("tool_name")
        trace["tool_name"] = tool_name

        args = decision.get("arguments",{})
        trace["tool_arguments"] = args

        if tool_name == "search_knowledge_base":
            query = args.get("query", request.message)
            top_k = args.get("top_k", 3)

            tool_result = search_vector_store(query=query, top_k=top_k)
            trace["tool_result"] = tool_result

            result = final_answer_with_tool_result(message=request.message, tool_result=tool_result)

            trace["answer"] = result
            trace["used_chunks"] = result.get("used_chunks")
            trace["duration_ms"] = int((time.time() - start_time) * 1000)

            write_agent_log(trace)

            return {
            "mode": "tool_call",
            "tool_name": tool_name,
            "tool_arguments": args,
            "contexts": tool_result,
            "answer": result.get("answer"),
            "used_chunks": result.get("used_chunks", []),
            "trace": trace
        }

        if tool_name == "list_chunks":
            limit = int(args.get("limit", 20))
            chunks = list_all_chunks(limit=limit)

            answer = f"当前返回了{len(chunks)}个chunk"

            trace["answer"] = answer
            trace["tool_result"] = chunks
            trace["duration_ms"] = int((time.time() - start_time) * 1000)

            write_agent_log(trace)

            return {
                "mode" : "tool_call",
                "tool_name" : tool_name,
                "tool_arguments" : args,
                "tool_result" : chunks,
                "answer" : f"当前知识库返回了{len(chunks)}个文件",
                "trace": trace
            }
        
        if tool_name == "summerize_text":
            text = args.get("text","")

            summary = summarize_note(text)
            answer = summary.get("summary","")

            trace["duration_ms"] = int((time.time() - start_time) * 1000)
            trace["answer"] = answer
            trace["tool_result"] = summary

            write_agent_log(trace)

            return {
                "mode" : "tool_call",
                "tool_name" : tool_name,
                "tool_arguments" : text,
                "tool_result" : summary,
                "answer" : summary.get(summary,""),
                "trace": trace
            }
        
        trace["error"] = f"Unknown tool: {tool_name}"
        trace["duration_ms"] = int((time.time() - start_time) * 1000)

        write_agent_log(trace)

        return {
            "error" : f"Unknown tool : {tool_name}",
            "trace": trace
        }
    except Exception as e:
        trace["error"] = str(e)
        trace["duration_ms"] = int((time.time() - start_time) * 1000)

        write_agent_log(trace)

        return {
            "error": str(e),
            "trace": trace
        }
    

@app.get("/agent_logs")
def get_agent_logs(limit : int = 20):
    logs = read_agent_logs(limit=limit)

    return {
        "count": len(logs),
        "logs": logs
    }

    
