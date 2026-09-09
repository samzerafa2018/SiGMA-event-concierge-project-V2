# SiGMA Event Concierge V2

A grounded AI assistant for a fictional SiGMA Malta 2026 event dataset.
Visitors can search the agenda, find speakers and exhibitors, state their
availability, and build a conflict-free itinerary.

This is an independent technical prototype created for a Junior AI Engineer
task. It is not an official SiGMA product. V2 is a separate post-submission
iteration; the original submission remains unchanged.

## What it does

- Finds sessions by title, topic, speaker, ID, day, time or room.
- Finds exhibitors by name, category, product or stand.
- Answers with agenda source citations such as `[S017]` and `[E001]`.
- Stores availability during the current conversation.
- Builds chronological, non-overlapping itineraries.
- Handles follow-ups such as Which AI sessions can I attend? and
  Build an agenda using those sessions.
- Rejects unsupported questions instead of inventing event information.

## How it works

Python performs structured filtering, retrieval, validation and itinerary
planning. Common factual questions can be answered locally without waiting for
a language model. Gemini or Ollama is used when natural-language synthesis is
helpful. Generated responses are checked against the retrieved agenda records;
if validation fails, the system uses a verified fallback.

The interface is built with Streamlit.

## Project structure

```text
SiGMA-event-concierge-project-V2/
├── app.py                              # Streamlit UI, conversation state and provider setup
├── mini_eval.py                        # Grounded end-to-end evaluation runner
├── requirements.txt                    # Python dependencies
├── .env.example                        # Safe provider configuration template
├── .gitignore                          # Excludes secrets, environments and caches
├── .streamlit/
│   └── config.toml                     # Streamlit theme configuration
├── data/
│   └── sigma_agenda.json               # Fictional sessions and exhibitors
├── concierge/
│   ├── __init__.py                     # Package initialisation
│   ├── answering.py                    # Answer routing, grounding, validation and fallbacks
│   ├── context.py                      # Conversation and availability context
│   ├── llm.py                          # Gemini/Ollama adapters and retry handling
│   ├── models.py                       # Pydantic agenda models and data loading
│   ├── planner.py                      # Availability filtering and itinerary planning
│   └── retrieval.py                    # Structured and weighted lexical retrieval
└── tests/
    ├── test_answering.py               # Answer grounding and fallback tests
    ├── test_conversation_availability.py # Multi-turn availability tests
    ├── test_eval.py                    # Evaluation-suite tests
    ├── test_llm.py                     # Provider and retry tests
    ├── test_models.py                  # Data-validation tests
    ├── test_planner.py                 # Availability and itinerary tests
    └── test_retrieval.py               # Retrieval tests
```

## Requirements

- Python 3.11 or newer
- Git
- A Gemini API key **or** a local Ollama installation

Gemini is the default and easiest option. API keys must remain in the local
`.env` file and must never be committed to Git.

## Run on Windows

Open PowerShell:

```powershell
git clone https://github.com/samzerafa2018/SiGMA-event-concierge-project-V2.git
cd .\SiGMA-event-concierge-project-V2

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Copy-Item .\.env.example .\.env
```

Open `.env`, add your own Gemini API key, and keep:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=replace_with_your_own_key
```

Start the application:

```powershell
streamlit run .\app.py
```

## Run on macOS or Linux

Open Terminal:

```bash
git clone https://github.com/samzerafa2018/SiGMA-event-concierge-project-V2.git
cd SiGMA-event-concierge-project-V2

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env
```

Open `.env`, add your own Gemini API key, and keep:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=replace_with_your_own_key
```

Start the application:

```bash
streamlit run app.py
```

Streamlit normally opens the application automatically at
`http://localhost:8501`.

## Use Ollama instead of Gemini

Install [Ollama](https://ollama.com/), then download the local model:

```bash
ollama pull llama3.2:3b
```

Set the following values in `.env`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

Ensure Ollama is running before starting Streamlit. A Gemini key is not needed
when Ollama is selected.

## Run the tests

With the virtual environment active:

```bash
python -m pytest -q
```

Run the end-to-end evaluation, including the configured model provider:

```bash
python mini_eval.py --with-llm
```

The latest verified run passed the complete automated test suite and all 17
grounded end-to-end evaluation cases.

## Other devices

The Python application runs locally on Windows, macOS or Linux. Once deployed
to a server or Streamlit hosting, the interface can be opened through a modern
browser on laptops, phones and tablets. A public deployment should use managed
secrets, rate limiting, monitoring, privacy controls and an authorised live
agenda source.

## Current scope

The included agenda is fictional and stored locally as JSON. The prototype does
not receive live schedule changes, persist accounts or itineraries, or include
the operational controls required for a public production launch.

If port `8501` is already in use, start Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```
