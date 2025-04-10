# 🤖 Agentic AI SQL Agent

A project scaffold to run an agentic SQL agent using **Python 3.12** and **Groq API**.

---

## 🛠️ Setup Instructions

Follow the steps below to set up and run the project in your local environment.

---

### 📦 Create Conda Environment

Make sure Conda is installed, then create and activate the environment:

```bash
conda create -n agentic_ai python=3.12
conda activate agentic_ai
conda install --yes --file requirements.txt
```

---

### 🔐 Set Up Environment Variables

1. Create a `.env` file in the root directory of the project.

2. Add the following variables to the `.env` file:

```env
GROQ_API_KEY=your_api_key
AGENTOPS_API_KEY=your_api_key
```

---

### 📂 Navigate to SQL Agent Directory

Change your working directory to `sql_agent`:

```bash
cd sql_agent
```

---

### 🚀 Running the Application

Run the agent using the following command:

```bash
uvicorn src.sql_agent.main:app --reload
```
<del> python -m src.sql_agent.main

> **Note:** Absolute imports are used in this project, so it's important to run the module exactly as shown above.

---

### 📁 Project Directory Structure

```plaintext
sql-agent/
│
├── requirements.txt
├── .env
├──  src/
│       └── sql_agent/
│           ├── main.py
│           └── crew.py
└── README.md
```

---

Happy building with Agentic AI! 🧠💻

### AGENT
### 🔍 Components Breakdown

- **User Input**:  
  Natural language query provided by the user.

- **SqlAgent**:  
  Orchestrates the entire workflow using CrewAI, coordinating agents and tasks.

- **Agents**:
  - `data_engineer`:  
    Uses `NL2SQLTool` to generate SQL queries from natural language and execute them.
  - `data_analyst`:  
    Analyzes the SQL output using Python (`CodeInterpreterTool`) and writes insights to a markdown file.

- **Tasks**:
  - `query_creation_task`:  
    Converts natural language into executable SQL queries.
  - `analysis_task`:  
    Interprets SQL results, performs analysis, and creates a report.

- **Tools**:
  - `NL2SQLTool`:  
    Converts NL → SQL and interacts with SQLite database.
  - `CodeInterpreterTool`:  
    Executes Python code for data analysis and visualization.
  - `FileWriterTool`:  
    Persists the final analysis output to a markdown report.

- **Output**:  
  Final markdown report containing insights and analysis is returned to the user.


### Workflow
![Workflow Diagram](agent_workflow.svg)

<details>
<summary>📊 Click to view Agentic Workflow Diagram</summary>

```mermaid
graph TD
    A["User Input (e.g., 'Analyze sales patterns for product B')"] --> B["crew.kickoff()"]
    B --> C["SqlAgent.crew()"]
    C --> D1["data_engineer Agent"]
    C --> D2["data_analyst Agent"]

    D1 --> E1["query_creation_task"]
    E1 --> F1["NL2SQLTool (Generate + Execute SQL) + FileWriterTool"]

    D2 --> E2["analysis_task"]
    E2 --> F2["CodeInterpreterTool + FileWriterTool"]

    F1 --> G["SQL DB (sales.db)"]
    F2 --> H["Final Report (Markdown)"]

    H --> I["Return Result to User"]
</details> ```
