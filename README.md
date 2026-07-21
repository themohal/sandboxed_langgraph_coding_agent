# 🤖 Agentic Daytona Sandbox with LangGraph & Streamlit

An advanced, human-in-the-loop AI coding companion built with a stateful **LangGraph ReAct architecture** and an isolated **Daytona Data Analysis Sandbox**. This application allows an LLM agent to safely write, test, debug, and execute code within a secure container environment while maintaining full human approval controls.

## 🚀 Key Features

*   **Agentic Feedback Loop**: The AI agent autonomously generates code, reads execution track logs, and self-heals syntax or runtime issues.
*   **Human-In-The-Loop (HITL)**: Custom LangGraph compilation breakpoints halt execution *before* any terminal processing, displaying the code block directly on screen for user vetting.
*   **Seamless Chat Rejections**: Users can natively submit text feedback inside a localized layout container using Streamlit `@st.fragment` cards. The graph clears its upcoming queue and replans without jarring page refreshes or input text stickiness.
*   **Persistent Sandboxing**: Configured with a static `sandbox_id` to ensure environmental variables, newly created datasets, or installed python packages remain preserved across multiple chat message intervals.

---

## 🛠️ Project Installation

Ensure you have Python 3.10+ installed locally on your machine.

1. **Clone the repository and navigate into the workspace folder:**
   ```bash
   git clone <your-repository-url>
   cd langraph_codeagent_with_daytona
   ```

2. **Initialize a secure virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the environment:**
   *   **Windows:** `.venv\Scripts\activate`
   *   **macOS / Linux:** `source .venv/bin/activate`

4. **Install necessary runtime requirements:**
   ```bash
   pip install streamlit python-dotenv langchain-openai langchain-daytona-data-analysis langgraph
   ```

---

## 🔑 Environment Configuration

Create a file named `.env` in the root of your project directory and populate it with your live credential strings:

```env
# OpenAI Engine Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Daytona Secure Sandbox Configuration
DAYTONA_API_KEY=your_daytona_api_key_here
```

---

## 🎯 How to Run & Test

Launch the Streamlit framework server straight from your active terminal:
```bash
streamlit run agent.py
```

### 🧪 Suggested Verification Profiles

*   **The Happy Path (Approval)**: Ask the agent, `"Calculate the 15th Fibonacci number."` Review the code inside the popup approval panel and click **Approve & Run Code**. The final computed answer will output to your timeline.
*   **The Feedback Loop (Rejection)**: Prompt the agent, `"Sort an array of 5 numbers using bubble sort."` Open the rejection popover card, enter `"Do not use bubble sort, use python's built-in sorted() instead"` and click **Submit Feedback**. The panel will immediately display a loading animation, clear out your old input text, and serve up a fresh `sorted()` script block for a second review cycle.
