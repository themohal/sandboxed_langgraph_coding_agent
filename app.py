import os
import streamlit as st
# from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_daytona_data_analysis import DaytonaDataAnalysisTool
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# # 1. Load Environmental Variables
# load_dotenv()

# 2. Page Configuration & Custom CSS Injection
st.set_page_config(
    page_title="Daytona AI Sandbox Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to eliminate "dry" default look
st.markdown("""
<style>
    /* Gradient Banner Header */
    .header-container {
        background: linear-gradient(135deg, #1e1e2f 0%, #0f172a 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    
    /* Action Approval Card */
    .approval-card {
        background-color: rgba(240, 246, 255, 0.5);
        border: 1px solid #cbd5e1;
        border-left: 5px solid #2563eb;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Sidebar Accent Badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-online { background-color: #dcfce7; color: #15803d; }
    .status-offline { background-color: #fee2e2; color: #b91c1c; }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Setup
with st.sidebar:
    st.image("https://github.com/themohal/sandboxed_langgraph_coding_agent/blob/main/logo.png?raw=true", width=180)
    st.title("Control Panel")
    st.caption("Agentic Execution & Environment Manager")
    st.divider()

    # Environment Check
    openai_key = os.environ.get("OPENAI_API_KEY")
    daytona_key = os.environ.get("DAYTONA_API_KEY")
    
    st.subheader("🔑 Environment Status")
    col_a, col_b = st.columns(2)
    with col_a:
        if openai_key:
            st.markdown('<span class="status-badge status-online">✓ OpenAI</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-offline">✗ OpenAI</span>', unsafe_allow_html=True)
    with col_b:
        if daytona_key:
            st.markdown('<span class="status-badge status-online">✓ Daytona</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-offline">✗ Daytona</span>', unsafe_allow_html=True)

    if not openai_key or not daytona_key:
        st.error("Missing required API keys inside your local `.env` file.")
        st.stop()
        
    st.divider()
    st.subheader("⚙️ Agent Settings")
    st.text_input("Active Model", value="gpt-4o", disabled=True)
    
    if st.button("🧹 Clear Chat & Reset Session", use_container_width=True):
        st.session_state.graph_messages = []
        if "graph_app" in st.session_state:
            del st.session_state["graph_app"]
        st.rerun()

# 4. Header Section
st.markdown("""
<div class="header-container">
    <h2 style="margin: 0; padding: 0; color: #ffffff;">⚡ Daytona AI Code Studio</h2>
    <p style="margin: 0.4rem 0 0 0; opacity: 0.8; font-size: 0.95rem;">
        An autonomous LangGraph agent operating safely inside a isolated Daytona sandbox environment with Human-In-The-Loop approval.
    </p>
</div>
""", unsafe_allow_html=True)

# 5. Initialize Graph State and Build Logic
if "graph_app" not in st.session_state:
    st.session_state.sandbox_id = "streamlit_shared_sandbox_session"
    
    try:
        daytona_tool = DaytonaDataAnalysisTool(sandbox_id=st.session_state.sandbox_id)
        st.session_state.daytona_ready = True
    except Exception as e:
        st.session_state.daytona_ready = False
        st.error(
            f"🚨 **Daytona Initialization Failed!**\n\n"
            f"**Error Details:** {str(e)}\n\n"
            f"👉 **How to Fix:** Visit [https://daytona.io](https://daytona.io) to clean up unused sandboxes and refresh."
        )
        st.stop()

    if st.session_state.daytona_ready:
        tools = [daytona_tool]
        model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

        def call_model(state: MessagesState):
            system_instruction = SystemMessage(
                content=(
                    "You are an autonomous software engineer operating inside a Daytona sandbox container. "
                    "If a tool execution returns an error, inspect the exception, fix the code immediately, "
                    "and issue a NEW tool call with the corrected script until execution succeeds."
                )
            )
            messages_with_system = [system_instruction] + state["messages"]
            return {"messages": [model.invoke(messages_with_system)]}

        def should_continue(state: MessagesState):
            last_message = state["messages"][-1]
            if getattr(last_message, "tool_calls", None):
                return "tools"
            return "__end__"

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")

        st.session_state.graph_app = workflow.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["tools"]
        )
        st.session_state.config = {"configurable": {"thread_id": "streamlit_session"}}

# 6. Initialize Chat Memory
if "graph_messages" not in st.session_state:
    st.session_state.graph_messages = []

# 7. Render Onboarding Empty State if Conversation is Blank
if not st.session_state.graph_messages:
    st.markdown("### 👋 Welcome! How can the agent help you today?")
    st.caption("Select a sample scenario below or type your request in the chat input.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("📊 **Data Analysis**")
            st.write("Generate synthetic sales data and plot a line chart using Pandas & Seaborn.")
            if st.button("Run Example 1", key="ex1", use_container_width=True):
                st.session_state.pending_prompt = "Generate synthetic monthly sales data for 2025, plot a chart using Seaborn, and save it as an image."
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("⚡ **Algorithm Execution**")
            st.write("Write and benchmark a Python script that calculates Fibonacci numbers.")
            if st.button("Run Example 2", key="ex2", use_container_width=True):
                st.session_state.pending_prompt = "Write a Python script comparing recursive vs dynamic programming Fibonacci execution speeds."
                st.rerun()

    with col3:
        with st.container(border=True):
            st.markdown("🌐 **System Diagnostics**")
            st.write("Inspect OS properties, CPU count, memory usage, and installed packages.")
            if st.button("Run Example 3", key="ex3", use_container_width=True):
                st.session_state.pending_prompt = "Inspect the system environment: display Python version, CPU core count, available memory, and disk space."
                st.rerun()

    st.divider()

# 8. Render Existing Chat Messages
for idx, msg in enumerate(st.session_state.graph_messages):
    if msg.type == "human":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)
            
    elif msg.type == "ai":
        if msg.content:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg.content)
                
        if msg.tool_calls:
            first_call = msg.tool_calls[0]
            code_run = first_call["args"].get("data_analysis_python_code", "# Code block unavailable")
            with st.chat_message("assistant", avatar="🤖"):
                with st.expander("🛠️ Proposed Daytona Execution Script", expanded=False):
                    st.code(code_run, language="python")
                    
    elif msg.type == "tool":
        with st.chat_message("assistant", avatar="⚙️"):
            if "Error" in msg.content or "Traceback" in msg.content:
                st.error(f"❌ **Execution Failed:**\n```python\n{msg.content}\n```")
            else:
                st.success(f"✅ **Execution Succeeded:**\n```\n{msg.content}\n```")

# 9. Core Stream Runner
def run_agent_stream(initial_input=None):
    app = st.session_state.graph_app
    config = st.session_state.config
    stream_generator = app.stream(initial_input, config, stream_mode="values")
    
    for chunk in stream_generator:
        st.session_state.graph_messages = chunk["messages"]

# 10. Process Input Prompts (from input bar or quickstart cards)
active_prompt = st.chat_input("Ask the agent to write code, solve problems, or analyze data...")

if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    del st.session_state["pending_prompt"]

if active_prompt:
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Agent is reasoning and writing code..."):
            run_agent_stream({"messages": [("user", active_prompt)]})
            st.rerun()

# 11. Interactive Control Panel Wrapper
@st.fragment
def render_interaction_controls():
    current_state = st.session_state.graph_app.get_state(st.session_state.config)
    
    if current_state.next and "tools" in current_state.next:
        last_ai_msg = current_state.values["messages"][-1]
        pending_tool_calls = last_ai_msg.tool_calls
        
        if pending_tool_calls and len(pending_tool_calls) > 0:
            first_call = pending_tool_calls[0]
            proposed_code = first_call["args"].get("data_analysis_python_code", "# No code found")
            tool_call_id = first_call.get("id")
        else:
            return

        panel_area = st.empty()
        
        with panel_area.container():
            st.markdown('<div class="approval-card">', unsafe_allow_html=True)
            st.subheader("🛡️ Human Control Gate: Code Execution Requested")
            st.markdown("The agent generated the following script and requires your authorization to run it inside the Daytona container:")
            
            st.code(proposed_code, language="python")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🚀 Authorize & Execute", type="primary", use_container_width=True):
                    panel_area.info("⏳ Executing approved script inside Daytona sandbox...")
                    run_agent_stream(None)
                    
                    new_state = st.session_state.graph_app.get_state(st.session_state.config)
                    if new_state.next and "tools" in new_state.next:
                        st.rerun()
                    else:
                        st.rerun(scope="app")
            with col2:
                with st.popover("⛔ Reject / Request Changes", use_container_width=True):
                    unique_input_key = f"feedback_input_{len(st.session_state.graph_messages)}"
                    feedback = st.text_input(
                        "Provide feedback to the agent:", 
                        placeholder="e.g., Use an iterative loop instead of recursion.", 
                        key=unique_input_key
                    )
                    
                    if st.button("Submit Rejection", type="primary", use_container_width=True):
                        rejection_text = f"User rejected execution. Instructions: {feedback}"
                        mock_tool_message = ToolMessage(content=rejection_text, tool_call_id=tool_call_id)
                        
                        st.session_state.graph_app.update_state(
                            st.session_state.config, 
                            {"messages": [mock_tool_message]}, 
                            as_node="tools"
                        )
                        
                        panel_area.warning("🔄 Rejection logged. Agent is revising plan...")
                        run_agent_stream(None)
                        st.rerun(scope="app")
            st.markdown('</div>', unsafe_allow_html=True)

# 12. Render Controls
if "graph_app" in st.session_state:
    render_interaction_controls()

