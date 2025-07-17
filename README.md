# Agentic Multi-Agent Research Assistant

This project is a Python-based multi-agent research assistant that leverages the `agno` framework and the Groq LLM API to answer complex queries using a team of specialized agents. Each agent is equipped with domain-specific tools (web search, finance, scientific literature, news, medical, Wikipedia, and website extraction) and collaborates to provide comprehensive, source-backed answers.

## Features

- **Multi-Agent Collaboration:** Agents with different roles (web, finance, scientific, news, medical, Wikipedia, website) work together to answer queries.
- **Domain-Specific Tools:** Integrates DuckDuckGo, Yahoo Finance, Arxiv, HackerNews, PubMed, Wikipedia, and website extraction tools.
- **Groq LLM Integration:** Uses the `llama-3.3-70b-versatile` model via the Groq API for advanced reasoning and language understanding.
- **Streaming Responses:** Supports streaming output for real-time feedback.
- **Environment Variable Management:** Loads sensitive API keys from a `.env` file using `python-dotenv`.

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd agentic
```

### 2. Install Dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

> **Note:** The `.env` file is loaded automatically by the script.

### 4. Run the Application

```bash
python app.py
```

## Usage

The main script (`app.py`) defines a team of agents, each with a specific domain and toolset. The team is tasked with answering a research question, for example:

```python
agent_team.print_response("explain me about recent advancements in pubmed", stream=True)
```

You can modify the prompt or the agent team composition as needed.

## Agents and Tools

- **Web Agent:** DuckDuckGo or Website extraction
- **Finance Agent:** Yahoo Finance (stock prices, recommendations, company info)
- **Scientific Agent:** Arxiv (scientific papers)
- **HackerNews Agent:** HackerNews (tech news)
- **Med Agent:** PubMed (medical literature)
- **Wiki Agent:** Wikipedia

Each agent uses the `Groq` LLM and is configured to include sources and/or display data in tables.

## Dependencies

Key dependencies (see `requirements.txt` for full list):

- `agno`
- `groq`
- `yfinance`
- `duckduckgo_search`
- `arxiv`
- `HackerNews`
- `python-dotenv`
- `pandas`
- `aiohttp`
- `requests`

## Customization

- **Add/Remove Agents:** Edit the `agent_team` list in `app.py`.
- **Change Prompts:** Modify the string passed to `agent_team.print_response`.
- **Add Tools:** Import and add new tools to agents as needed.

## Troubleshooting

- **API Key Issues:** Ensure your `.env` file is present and contains a valid `GROQ_API_KEY`.
- **Dependency Errors:** Run `pip install -r requirements.txt` to ensure all packages are installed.
- **Model/Tool Errors:** Some tools may require additional setup or have usage limits.

## License

[MIT License](LICENSE) (add your license file if needed) 