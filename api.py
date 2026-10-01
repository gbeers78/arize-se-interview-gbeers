from dotenv import load_dotenv
from fastapi import FastAPI
from phoenix.otel import register
from pydantic import BaseModel

load_dotenv()

register(project_name="se-interview", auto_instrument=True)

# Import LangChain and LangGraph after tracing is registered.
from langchain_core.messages import HumanMessage

from agent import build_agent

agent = build_agent()

app = FastAPI(
    title="LangGraph Agent API",
    description="A simple API for interacting with a LangGraph agent that can search the web",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message to the agent and get a response."""
    result = agent.invoke({"messages": [HumanMessage(content=request.message)]})
    return ChatResponse(response=result["messages"][-1].content)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
