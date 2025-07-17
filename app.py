from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.arxiv import ArxivTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.pubmed import PubmedTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.website import WebsiteTools
from dotenv import load_dotenv
import os

load_dotenv()


web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions="Use tables to display data",
    markdown=True,
)

scientific_agent = Agent(
    name = "Scientific Agent",
    role = "Get scientific data",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [ArxivTools()],
    instructions = "Always include sources",
    markdown = True,
)

hacker_news_agent = Agent(
    name="Hackernews Team",
    role = "Get hackernews data",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools=[HackerNewsTools()],
    show_tool_calls=True,
    markdown=True,
)
med_agent = Agent(
    name = "Med Agent",
    role = "Get medical data",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools=[PubmedTools()],
    show_tool_calls=True,
)

wiki_agent = Agent(
    name = "Wiki Agent",
    role = "Get wikipedia data",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools=[WikipediaTools()],
    show_tool_calls=True,
)

web_agent = Agent(
    name = "Web Agent",
    role = "Get web data",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools=[WebsiteTools()],
    show_tool_calls=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent, scientific_agent, hacker_news_agent, med_agent, wiki_agent,wiki_agent,web_agent],
    model=Groq(id="llama-3.3-70b-versatile"),  # You can use a different model for the team leader agent
    instructions=["Always include sources", "Use tables to display data"],
    # show_tool_calls=True,  # Uncomment to see tool calls in the response
    markdown=True,
)

# Give the team a task
agent_team.print_response("explain me about recent advancements in pubmed", stream=True)