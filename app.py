import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_daytona_data_analysis import DaytonaDataAnalysisTool
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 1. Load Environmental Variables from .env File
load_dotenv()

# 2. Page Configuration
st.set_page_config(page_title="Agentic Daytona Sandbox", page_icon="🤖", layout="wide")
st.title("🤖 LangGraph Code Agent with Daytona Sandbox")

# 3. Check for Required Environment Variables
if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("DAYTONA_API_KEY"):
    st.error("❌ Missing environment variables! Please ensure both `OPENAI_API_KEY` and `DAYTONA_API_KEY` are populated inside your local `.env` file.")
    st.stop()

# 4. Initialize Graph State and Build Core Workspace Logic Securely
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
            f"👉 **How to Fix:** Your cloud account has hit its storage limit. Please visit "
            f"[https://daytona.io](https://daytona.io) to delete or archive old, "
            f"unused sandbox sessions, then refresh this browser tab."
        )
        st.stop()

    if st.session_state.daytona_ready:
        tools = [daytona_tool]
        model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

        def call_model(state: MessagesState):
            system_instruction = SystemMessage(
                content=(
                    "You are an autonomous software agent operating inside a Daytona sandbox container. "
                    "If a tool execution returns an error or traceback, your absolute priority is to inspect "
                    "the exception, fix the code immediately, and issue a NEW tool call with the corrected script. "
                    "Do not just explain the error in text format. You must rewrite the code and execute it until it succeeds."
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

# Initialize graph historical message state track list if not present
if "graph_messages" not in st.session_state:
    st.session_state.graph_messages = []

# 5. Render Conversation History UI Components
# FIX: Map state list histories directly to capture paired execution code strings and terminal logs
for idx, msg in enumerate(st.session_state.graph_messages):
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
            
    elif msg.type == "ai":
        # Render standard text reasoning from the assistant if it exists
        if msg.content:
            with st.chat_message("assistant"):
                st.markdown(msg.content)
                
        # If this AI message triggered code execution, render it as an expanding block
        if msg.tool_calls:
            first_call = msg.tool_calls[0]
            code_run = first_call["args"].get("data_analysis_python_code", "# Code block unavailable")
            with st.chat_message("assistant"):
                with st.expander("💻 Review Code Executed Inside Daytona", expanded=False):
                    st.code(code_run, language="python")
                    
    elif msg.type == "tool":
        # Find and present the output logs right below the expanding code card block
        with st.chat_message("assistant"):
            if "Error" in msg.content or "Traceback" in msg.content:
                st.error(f"❌ **Daytona Sandbox Run Error:**\n```python\n{msg.content}\n```")
            else:
                st.success(f"💻 **Daytona Sandbox Console Output:**\n```\n{msg.content}\n```")

# 6. Core Agent Stream Handler
def run_agent_stream(initial_input=None):
    """Streams graph updates and syncs the output back to the Streamlit UI."""
    app = st.session_state.graph_app
    config = st.session_state.config
    
    stream_generator = app.stream(initial_input, config, stream_mode="values")
    
    for chunk in stream_generator:
        # Save structural graph historical message data types to sync memory state bindings
        st.session_state.graph_messages = chunk["messages"]

# 7. Capture New User Input Prompt
if prompt := st.chat_input("Ask the agent to code or analyze data..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Agent is planning workflow..."):
            run_agent_stream({"messages": [("user", prompt)]})
            st.rerun()

# 8. Self-Contained Fragment Interactive Panel Wrapper
@st.fragment
def render_interaction_controls():
    """Hides execution and feedback state-flows inside an isolated layout frame."""
    current_state = st.session_state.graph_app.get_state(st.session_state.config)
    
    if current_state.next and "tools" in current_state.next:
        last_ai_msg = current_state.values["messages"][-1]
        pending_tool_calls = last_ai_msg.tool_calls
        
        if pending_tool_calls and len(pending_tool_calls) > 0:
            first_call = pending_tool_calls[0]
            proposed_code = first_call["args"].get("data_analysis_python_code", "# No code argument found")
            tool_call_id = first_call.get("id")
        else:
            return

        panel_area = st.empty()
        
        with panel_area.container():
            st.info("⚠️ **Human Approval Required**: The agent is attempting to execute code inside Daytona.")
            st.code(proposed_code, language="python")
            
            col1, col2 = col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve & Run Code", type="primary", use_container_width=True):
                    panel_area.info("⏳ Running approved script inside Daytona sandbox...")
                    
                    # 1. Execute the approved container task step
                    run_agent_stream(None)
                    
                    # 2. Check if a runtime crash occurred and triggered a self-healing iteration loop
                    new_state = st.session_state.graph_app.get_state(st.session_state.config)
                    if new_state.next and "tools" in new_state.next:
                        st.rerun()
                    else:
                        st.rerun(scope="app")
            with col2:
                with st.popover("❌ Reject & Send Feedback", use_container_width=True):
                    unique_input_key = f"feedback_input_{len(st.session_state.graph_messages)}"
                    feedback = st.text_input("Why are you rejecting this code?", placeholder="e.g., Change the algorithm...", key=unique_input_key)
                    
                    if st.button("Submit Feedback", type="primary", use_container_width=True):
                        rejection_text = f"Rejection details: {feedback}. Please rewrite the solution."
                        
                        mock_tool_message = ToolMessage(content=rejection_text, tool_call_id=tool_call_id)
                        st.session_state.graph_app.update_state(
                            st.session_state.config, 
                            {"messages": [mock_tool_message]}, 
                            as_node="tools"
                        )
                        
                        panel_area.warning("🔄 Rejection sent! Replanning code locally inside chat...")
                        run_agent_stream(None)
                        st.rerun(scope="app")

# 9. Invoke the Interactive Layout Frame Block
if "graph_app" in st.session_state:
    render_interaction_controls()
