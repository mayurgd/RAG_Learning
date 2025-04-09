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
python -m src.sql_agent.main
```

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
