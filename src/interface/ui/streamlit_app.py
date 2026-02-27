"""
Streamlit UI — Industrial AI Knowledge Assistant Frontend.

Professional interface for the industrial automation chatbot
with structured response display, confidence indicators, and
safety warnings.
"""

from __future__ import annotations

import streamlit as st
import os
import sys
from pathlib import Path

# Ensure src is on the path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Settings
from src.application.chat_service import ChatService
from src.domain import ChatRequest, IndustrialDomain, RiskLevel, ConfidenceLevel


# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Industrial AI Knowledge Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── State Initialization ─────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialize Streamlit session state."""
    defaults = {
        "messages": [],
        "conversation_id": None,
        "chat_service": None,
        "rag_initialized": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_chat_service() -> ChatService | None:
    """Get or create the ChatService singleton."""
    if st.session_state.get("chat_service") is not None:
        return st.session_state["chat_service"]

    api_key = st.session_state.get("api_key", "")
    if not api_key:
        return None

    try:
        settings = Settings(
            openai_api_key=api_key,
            llm_model=st.session_state.get("model_choice", "gpt-3.5-turbo"),
            llm_temperature=st.session_state.get("temperature", 0.3),
        )
        service = ChatService(settings=settings)
        st.session_state["chat_service"] = service
        return service
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the sidebar with configuration options."""
    st.sidebar.markdown("# 🏭 Industrial AI Assistant")
    st.sidebar.markdown("---")

    # API Key
    st.sidebar.markdown("### Configuration")
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        key="api_key",
        help="Required for AI responses. Get one at platform.openai.com",
    )

    if api_key:
        st.sidebar.success("API Key configured")
    else:
        st.sidebar.warning("Enter your OpenAI API key to enable AI features")

    # Model selection
    st.sidebar.selectbox(
        "LLM Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        key="model_choice",
    )

    st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        key="temperature",
        help="Lower = more focused, Higher = more creative",
    )

    # RAG Toggle
    st.sidebar.checkbox(
        "Enable RAG (Document Context)",
        value=True,
        key="use_rag",
        help="Retrieve relevant context from industrial documentation",
    )

    # Initialize RAG button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Initialize RAG Pipeline", use_container_width=True):
        service = get_chat_service()
        if service:
            with st.spinner("Loading industrial documents..."):
                success = service.initialize_rag()
                if success:
                    st.session_state["rag_initialized"] = True
                    st.sidebar.success("RAG pipeline initialized!")
                else:
                    st.sidebar.error("RAG initialization failed. Check documents directory.")
        else:
            st.sidebar.error("Configure API key first.")

    # New Chat
    st.sidebar.markdown("---")
    if st.sidebar.button("🆕 New Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["conversation_id"] = None
        st.session_state["chat_service"] = None
        st.rerun()

    # Domain hints
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Example Queries")
    examples = [
        "How do I implement a PID control loop in a PLC?",
        "What are BACnet vs KNX protocol differences?",
        "Diagnose a chiller that's short-cycling",
        "Best practices for SCADA cybersecurity?",
        "How to set up predictive maintenance for pumps?",
    ]
    for ex in examples:
        st.sidebar.markdown(f"- _{ex}_")


# ── Response Rendering ────────────────────────────────────────────────────────

def render_industrial_response(response) -> None:
    """Render a structured industrial response with metadata."""
    # Main answer
    st.markdown(response.answer)

    # Metadata bar
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        confidence_colors = {
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🟠",
            ConfidenceLevel.UNCERTAIN: "🔴",
        }
        icon = confidence_colors.get(response.confidence, "⚪")
        st.metric("Confidence", f"{icon} {response.confidence_score:.0%}")

    with col2:
        risk_colors = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🟠",
            RiskLevel.CRITICAL: "🔴",
        }
        icon = risk_colors.get(response.risk_level, "⚪")
        st.metric("Risk Level", f"{icon} {response.risk_level.value.upper()}")

    with col3:
        st.metric("Domain", response.domain.value)

    with col4:
        st.metric("Response Time", f"{response.response_time_ms:.0f}ms")

    # Safety warnings
    if response.safety_warnings:
        st.markdown("---")
        for warning in response.safety_warnings:
            icon = "🔴" if warning.level in [RiskLevel.CRITICAL, RiskLevel.HIGH] else "⚠️"
            st.warning(f"{icon} **Safety Warning:** {warning.message}")

    # Sources
    if response.sources:
        with st.expander(f"📚 Sources ({len(response.sources)})"):
            for i, source in enumerate(response.sources):
                st.markdown(
                    f"**[Source {i + 1}]** {source.document} "
                    f"(relevance: {source.relevance_score:.2f})"
                )
                st.markdown(f"_{source.excerpt[:150]}..._")

    # Hallucination flags
    if response.hallucination_flags:
        with st.expander("⚠️ Quality Notes"):
            for flag in response.hallucination_flags:
                st.markdown(f"- {flag}")


# ── Main Chat Interface ──────────────────────────────────────────────────────

def render_chat() -> None:
    """Render the main chat interface."""
    st.markdown("# 🏭 Industrial AI Knowledge Assistant")
    st.markdown(
        "_AI-powered assistant for PLC programming, SCADA systems, "
        "building automation, and industrial troubleshooting_"
    )
    st.markdown("---")

    # Display chat history
    for msg in st.session_state["messages"]:
        role = msg["role"]
        with st.chat_message("user" if role == "user" else "assistant"):
            if role == "user":
                st.markdown(msg["content"])
            else:
                # If it's a structured response, render it
                if "response_obj" in msg:
                    render_industrial_response(msg["response_obj"])
                else:
                    st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about industrial automation..."):
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        service = get_chat_service()

        with st.chat_message("assistant"):
            if service:
                with st.spinner("Analyzing..."):
                    request = ChatRequest(
                        message=prompt,
                        conversation_id=st.session_state.get("conversation_id"),
                        use_rag=st.session_state.get("use_rag", True),
                    )
                    result = service.process_message(request)
                    st.session_state["conversation_id"] = result.conversation_id

                    render_industrial_response(result.response)

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": result.response.answer,
                        "response_obj": result.response,
                    })
            else:
                fallback = (
                    "Please configure your OpenAI API key in the sidebar "
                    "to enable AI-powered responses."
                )
                st.markdown(fallback)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": fallback,
                })


# ── Main Entry ────────────────────────────────────────────────────────────────

def main() -> None:
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
