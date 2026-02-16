"""
Faraday Web Research Agent – Streamlit Interface
===============================================
A Streamlit UI for interacting with the Web Research Agent.
Run with: streamlit run app.py
"""
from __future__ import annotations

import os, json, colorsys, textwrap, asyncio, re, time # Removed requests, time; Added asyncio
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie  # Animated loaders

# Added imports for agent and schemas
from research_system.agent import run_web_research
from research_system.schemas import ResearchReport, ErrorResponse



# ────────────────────────────
# Configuration (inline) 🛠️
# ────────────────────────────
LOGO_PATH: str = os.getenv("AGENT_LOGO", "Logo.png")
PRIMARY_COLOR = "#4D96FF"  # Accent color
BG_COLOR = "#0E1117"
BG_SECONDARY = "#1B1E24"
TEXT_COLOR = "#FAFAFA"
FONT_FAMILY = "Inter, sans-serif"
LOADER_URL = "https://assets5.lottiefiles.com/private_files/lf30_editor_46utqktq.json" # Spinner animation URL
MAX_SUMMARY_WORDS = 150

# ────────────────────────────
# Helper functions
# ────────────────────────────

def hls_to_hex(hue: float, light: float = 0.5, sat: float = 0.8) -> str:
    """Convert HLS color values to a hex string."""
    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def source_color(tool_name: Optional[str]) -> str:
    """Generate a consistent color based on the tool name hash."""
    if not tool_name:
        return PRIMARY_COLOR
    # Simple hash-based color generation for visual distinction
    hue = hash(tool_name) % 360 / 360.0
    return hls_to_hex(hue, light=0.6, sat=0.7)

@st.cache_data(show_spinner=False)
def load_lottie(url: str) -> dict | None:
    """Fetch a Lottie animation and cache it."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def truncate(text: str, words: int = MAX_SUMMARY_WORDS) -> str:
    """Truncate text to a specified number of words."""
    if not text: return ""
    parts = text.split()
    return text if len(parts) <= words else " ".join(parts[:words]) + " …"

def render_report(report_data: Dict[str, Any]):
    """Renders the ResearchReport data."""
    if not report_data:
        st.error("No report data received.")
        return

    def is_empty_section(text: str) -> bool:
        if not text:
            return True
        cleaned = " ".join(text.split()).lower()
        if len(cleaned) < 20:
            return True
        return cleaned in {"no content.", "no content", "n/a", "none"}

    def is_noise_source(title: str, snippet: Optional[str]) -> bool:
        noise_terms = [
            "read more",
            "learn more",
            "click here",
            "sign in",
            "sign up",
            "subscribe",
            "contact us",
            "privacy policy",
            "terms of service",
            "cookie",
        ]
        title_clean = (title or "").strip().lower()
        snippet_clean = (snippet or "").strip().lower()
        if len(title_clean) < 4:
            return True
        return any(term in title_clean or term in snippet_clean for term in noise_terms)

    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Remove bracketed ellipses like "[...]" or "[ ... ]".
        return re.sub(r"\[\s*\.\.\.\s*\]", "", text)

    query = report_data.get("query", "N/A")
    summary = clean_text(report_data.get("summary", "No summary provided."))
    sections = report_data.get("sections", [])
    sources = report_data.get("sources", [])
    biases = clean_text(report_data.get("potential_biases") or "")

    st.subheader(f"Research Report for: \"{query}\"")

    # --- Display Summary ---
    st.markdown("### Executive Summary")
    st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)

    # --- Display Sections ---
    if sections:
        st.markdown("### Detailed Findings")
        shown_sections = 0
        for section in sections:
            heading = section.get('heading', 'Section')
            content = clean_text(section.get('content', ''))
            heading_lower = heading.strip().lower()
            if heading_lower in {"executive summary", "summary", "overview"}:
                continue
            if is_empty_section(content):
                continue
            st.markdown(f"#### {heading}")
            st.markdown(content, unsafe_allow_html=True)
            shown_sections += 1
        if shown_sections == 0:
            st.info("No detailed sections were generated in the report.")
    else:
        st.info("No detailed sections were generated in the report.")

    # --- Display Potential Biases/Limitations ---
    if biases:
        st.markdown("### Potential Biases & Limitations")
        st.warning(biases)

    # --- Display Sources ---
    if sources:
        st.markdown("### Sources Consulted")
        link_lines = []
        for index, src in enumerate(sources, start=1):
            url = src.get("url")
            if not url:
                continue
            title = src.get("title") or url
            tool_used = src.get("tool_used")
            snippet = clean_text(src.get("snippet") or "")
            if is_noise_source(title, snippet):
                continue
            tool_suffix = f" — {tool_used}" if tool_used and tool_used != "tavily_search" else ""
            line = f"{index}. [{title}]({url}){tool_suffix}"
            if snippet:
                line += f"\n    {snippet}"
            link_lines.append(line)
        if link_lines:
            st.markdown("\n".join(link_lines))
        else:
            st.info("No source links were available in the report.")
    else:
        st.info("No sources were listed in the final report.")


def is_overloaded_error(message: str) -> bool:
    if not message:
        return False
    message_lower = message.lower()
    return "overloaded" in message_lower or "error code: 529" in message_lower


# ────────────────────────────
# Global page settings
# ────────────────────────────
st.set_page_config(
    page_title="Web Research Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Web Research Agent - An AI assistant to research topics online."
    }
)

# ────────────────────────────
# Custom CSS styling
# ────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400,0,0');

    /* Root variables for theming */
    :root {{
        --primary-color: {PRIMARY_COLOR};
        --text-color: {TEXT_COLOR};
        --background-color: {BG_COLOR};
        --secondary-background-color: {BG_SECONDARY};
        --font: {FONT_FAMILY};
    }}

    /* Base styling */
    html, body, [class*="st"] {{
        font-family: var(--font);
    }}

    .material-symbols-outlined {{
        font-family: 'Material Symbols Outlined' !important;
        font-variation-settings: 'opsz' 24, 'wght' 400, 'FILL' 0, 'GRAD' 0;
    }}

    .material-icons {{
        font-family: 'Material Icons' !important;
        font-weight: normal;
        font-style: normal;
        font-size: 20px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
    }}

    .stApp {{
        background-color: var(--background-color);
        color: var(--text-color);
    }}

    a {{
        color: var(--primary-color);
    }}

    /* Logo size adjustment */
    .logo-container img {{
        width: auto !important;
        height: 80px !important;
    }}

    /* Search bar styling */
    .stTextInput > div > div > input {{
        text-align: center;
        font-size: 1.25em;
        background-color: {BG_SECONDARY};
        color: white;
        border-radius: 25px;
        border: 1px solid {PRIMARY_COLOR};
        padding: 12px 20px;
    }}

    /* Main container */
    .main-container {{
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }}

    /* Logo container */
    .logo-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
        margin-top: 30px;
    }}

    /* Summary box */
    .summary-box {{
        background-color: {BG_SECONDARY};
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #333;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        line-height: 1.6;
    }}

    /* Loading steps */
    .loader-step {{
        display: flex;
        align-items: center;
        margin: 10px 0;
        padding: 12px 15px;
    }}

    .loader-step-active {{
        border-left: 3px solid {PRIMARY_COLOR};
        box-shadow: 0 0 8px {PRIMARY_COLOR}40;
    }}

    .loader-step-complete {{
        border-left: 3px solid #00CC66;
        box-shadow: 0 0 8px #00CC6640;
    }}

    .loader-icon {{
        margin-right: 15px;
        font-size: 18px;
    }}

    /* Additional tweaks */
    h1, h2, h3 {{
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }}

    .stApp > header {{
        background-color: transparent;
    }}

    /* Remove expander icon text that overlaps headings */
    div[data-testid="stExpander"] summary svg,
    div[data-testid="stExpander"] summary .material-icons,
    div[data-testid="stExpander"] summary .material-symbols-outlined {{
        display: none !important;
    }}
    div[data-testid="stExpander"] summary {{
        padding-left: 0.25rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────
# Main app layout
# ────────────────────────────
# Header with logo
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
if os.path.exists(LOGO_PATH):
    try:
        logo_image = Image.open(LOGO_PATH)
        st.image(logo_image, width=900, output_format="PNG", use_container_width=False, caption="")
    except Exception as e:
        st.error(f"Error loading logo: {e}")
        st.markdown("<h1 style='text-align:center;margin-bottom:0'>Faraday Web Research Agent</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1 style='text-align:center;margin-bottom:0'>Faraday Web Research Agent</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;color:#888;margin-top:4px'>Your AI Research Assistant</h4>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Research query input bar
query_input = st.text_input(
    "Research Query",
    placeholder="Enter your research query...",
    label_visibility="collapsed"
)

market_research_enabled = st.checkbox("Market Research", value=False)

# Initialize session state for tracking progress
if 'progress_state' not in st.session_state:
    st.session_state.progress_state = {
        'status': 'idle', # idle, running, completed, error
        'error_message': None,
        'report_data': None, # Renamed from 'api_data'
        'current_query': None # Store the query associated with the current state
    }

# Results container
results_container = st.container()

if query_input:
    with results_container:
        current_status = st.session_state.progress_state['status']
        stored_query = st.session_state.progress_state.get('current_query')

        # --- Check if the query has changed --- NEW LOGIC
        if query_input != stored_query:
            # User entered a new query, reset state and start running
            st.session_state.progress_state['status'] = 'running'
            st.session_state.progress_state['error_message'] = None
            st.session_state.progress_state['report_data'] = None
            st.session_state.progress_state['current_query'] = query_input
            st.rerun()
        # --- END NEW LOGIC ---

        # Proceed with existing logic only if the query HAS NOT changed
        # --- Start the process if status is idle --- (This case might become less frequent)
        elif current_status == 'idle':
            # This could happen on the very first run after initial load
            st.session_state.progress_state['status'] = 'running'
            st.session_state.progress_state['error_message'] = None
            st.session_state.progress_state['report_data'] = None
            st.session_state.progress_state['current_query'] = query_input # Store query when starting
            st.rerun() # Trigger rerun to show spinner and start processing

        # --- Run the research directly if status is running ---
        elif current_status == 'running':
            # Show animated steps while running
            st.markdown('<div style="margin: 30px 0;">', unsafe_allow_html=True)
            step1_class = "loader-step loader-step-active" # Now active
            st.markdown(f'<div class="{step1_class}"><span class="loader-icon">🧠</span> Thinking & Researching...</div>', unsafe_allow_html=True)
            step2_class = "loader-step" # Not active yet
            st.markdown(f'<div class="{step2_class}"><span class="loader-icon">📄</span> Gathering & Synthesizing Information...</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Use spinner while the agent runs
            with st.spinner("Performing web research... This may take a few moments."):
                max_retries = 2
                base_delay_seconds = 6
                retry_notice = st.empty()

                for attempt in range(max_retries + 1):
                    try:
                        # Directly call the agent function
                        research_result = asyncio.run(
                            run_web_research(
                                query=query_input,
                                config={"market_research": market_research_enabled},
                            )
                        )

                        # Check the result type and update state
                        if isinstance(research_result, ResearchReport):
                            st.session_state.progress_state['status'] = 'completed'
                            st.session_state.progress_state['report_data'] = research_result.dict() # Store report as dict
                        elif isinstance(research_result, ErrorResponse):
                            details = f"{research_result.error}: {research_result.details}"
                            if is_overloaded_error(details) and attempt < max_retries:
                                delay_seconds = base_delay_seconds * (attempt + 1)
                                retry_notice.info(
                                    f"The model is overloaded. Retrying in {delay_seconds} seconds..."
                                )
                                time.sleep(delay_seconds)
                                continue
                            st.session_state.progress_state['status'] = 'error'
                            st.session_state.progress_state['error_message'] = details
                        else:
                            # Handle unexpected return type
                            st.session_state.progress_state['status'] = 'error'
                            st.session_state.progress_state['error_message'] = f"Agent returned an unexpected result type: {type(research_result)}"

                        st.rerun() # Rerun to display results or error

                    except ImportError as ie:
                        st.session_state.progress_state['status'] = 'error'
                        st.session_state.progress_state['error_message'] = f"Import Error: {ie}. Ensure agent components are installed and accessible."
                        st.rerun()
                    except Exception as e:
                        if is_overloaded_error(str(e)) and attempt < max_retries:
                            delay_seconds = base_delay_seconds * (attempt + 1)
                            retry_notice.info(
                                f"The model is overloaded. Retrying in {delay_seconds} seconds..."
                            )
                            time.sleep(delay_seconds)
                            continue
                        st.session_state.progress_state['status'] = 'error'
                        st.session_state.progress_state['error_message'] = f"An unexpected error occurred during research: {e}"
                        st.rerun()

        # --- Display results if process is complete ---
        elif current_status == 'completed':
            report_data = st.session_state.progress_state.get('report_data') # Use renamed key
            if report_data:
                 render_report(report_data)

                 # Reset progress state if user wants to search again - REMOVED BUTTON
                 # if st.button("New Research Query", use_container_width=True, type="primary"):
                 #     st.session_state.progress_state = {
                 #         'status': 'idle',
                 #         'error_message': None,
                 #         'report_data': None,
                 #         'current_query': None
                 #     }
                 #     st.rerun()
                 st.info("Enter a new query above to start another research task.") # Inform user

            else:
                 st.error("Completed status reached but no report data found.")
                 # Reset to allow trying again
                 st.session_state.progress_state['status'] = 'idle'
                 st.session_state.progress_state['current_query'] = None # Clear query too
                 st.rerun()

        # --- Display error if status is error ---
        elif current_status == 'error':
            st.error(f"Research failed: {st.session_state.progress_state.get('error_message', 'Unknown error')}")
            # Allow user to try again - REMOVED BUTTON
            # if st.button("Try New Research", use_container_width=True, type="primary"):
            #     st.session_state.progress_state = {
            #         'status': 'idle',
            #         'error_message': None,
            #         'report_data': None,
            #         'current_query': None
            #     }
            #     st.rerun()
            st.info("Enter a new query above to try again or start a new research task.") # Inform user

else:
    # Show prompt if no query is entered
    with results_container:
        st.markdown("""
        <div style="text-align: center; margin-top: 50px; color: #AAAAAA;">
            <h3>Enter a research query above to start</h3>
            <p>Example: "What are the pros and cons of universal basic income?"</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("</div>", unsafe_allow_html=True)  # Close main container
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #888; font-size: 0.8em;">
    <p>Faraday Web Research Agent • Powered by AI</p>
</div>
""", unsafe_allow_html=True)
