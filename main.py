from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.arxiv import ArxivTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.pubmed import PubmedTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.website import WebsiteTools

load_dotenv()

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Agent setup (copied from app.py)
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions="Use tables to display data",
    markdown=True,
)

scientific_agent = Agent(
    name = "Scientific Agent",
    role = "Get scientific data",
    model = OpenAIChat(id = "gpt-4o-mini"),
    tools = [ArxivTools()],
    instructions = "Always include sources",
    markdown = True,
)

hacker_news_agent = Agent(
    name="Hackernews Team",
    role = "Get hackernews data",
    model = OpenAIChat(id = "gpt-4o-mini"),
    tools=[HackerNewsTools()],
    show_tool_calls=True,
    markdown=True,
)
med_agent = Agent(
    name = "Med Agent",
    role = "Get medical data",
    model = OpenAIChat(id = "gpt-4o-mini"),
    tools=[PubmedTools()],
    show_tool_calls=True,
)

wiki_agent = Agent(
    name = "Wiki Agent",
    role = "Get wikipedia data",
    model = OpenAIChat(id = "gpt-4o-mini"),
    tools=[WikipediaTools()],
    show_tool_calls=True,
)

web_agent2 = Agent(
    name = "Web Agent",
    role = "Get web data",
    model = OpenAIChat(id = "gpt-4o-mini"),
    tools=[WebsiteTools()],
    show_tool_calls=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent, scientific_agent, hacker_news_agent, med_agent, wiki_agent, wiki_agent, web_agent2],
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=["Always include sources", "Use tables to display data"],
    markdown=True,
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = agent_team.run(request.message)
        # If result is an object with 'content', return that, else return as string
        response = getattr(result, 'content', result)
        return {"response": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    with open(os.path.join("static", "page.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content) 