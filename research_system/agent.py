import operator
import logging
import uuid
import json
import os
from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from pydantic import BaseModel, Field

from .config import get_primary_llm
from .tools import create_agent_tools
from .prompts import (
    AGENT_SYSTEM_PROMPT,
    REPORT_SYNTHESIS_TEMPLATE,
    MARKET_RESEARCH_SYSTEM_PROMPT,
    MARKET_REPORT_SYNTHESIS_TEMPLATE,
    report_parser,
)
from .schemas import ResearchReport, Source, IntermediateStep, ResearchRequest, ErrorResponse
from .postprocessor import improve_report
from langchain_core.exceptions import OutputParserException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Represents the state of the Web Research Agent."""
    query: str
    research_id: str
    market_research: bool
    messages: Annotated[Sequence[BaseMessage], operator.add]
    intermediate_steps: Annotated[List[tuple[ToolInvocation, Any]], operator.add] = []
    final_result: Optional[ResearchReport] = None


# --- Agent Nodes ---

def agent_node(state: AgentState, agent, tools, name: str):
    """Node that calls the agent model to decide the next action."""
    logger.info(f"[{name} - ID: {state.get('research_id')}] Agent node executing.")
    system_prompt = MARKET_RESEARCH_SYSTEM_PROMPT if state.get("market_research") else AGENT_SYSTEM_PROMPT
    system_prompt_content = system_prompt.format(
        tool_descriptions="\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    )

    messages_for_llm = [SystemMessage(content=system_prompt_content)] + list(state["messages"])

    logger.info(f"[{name} - ID: {state.get('research_id')}] Invoking agent LLM with {len(messages_for_llm)} total messages...")

    result: BaseMessage = agent.invoke(messages_for_llm)

    logger.info(f"[{name} - ID: {state.get('research_id')}] Agent LLM raw response type: {type(result)}")

    output_messages = [result] if isinstance(result, BaseMessage) else []
    if not output_messages:
         logger.warning(f"[{name} - ID: {state.get('research_id')}] Agent LLM did not return a valid BaseMessage. Result: {result}")

    return {"messages": output_messages}


def tool_node(state: AgentState, tool_executor, name: str):
    """Node that executes the tool chosen by the agent."""
    logger.info(f"[{name} - ID: {state.get('research_id')}] Tool node executing.")
    messages = state["messages"]
    if not messages:
        logger.warning(f"[{name} - ID: {state.get('research_id')}] No messages found in state. Cannot execute tool.")
        return {"messages": [], "intermediate_steps": state.get("intermediate_steps", [])}

    last_message = messages[-1]

    # Check if the last message is an AIMessage with tool_calls
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.info(f"[{name} - ID: {state.get('research_id')}] Last message is not an AIMessage with tool calls. Type: {type(last_message)}. Content: {getattr(last_message, 'content', '')[:100]}...")
        return {"messages": [], "intermediate_steps": state.get("intermediate_steps", [])}

    tool_invocation_list = []
    tool_messages = []
    intermediate_steps_updates = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_input = tool_call.get("args", {})
        tool_call_id = tool_call.get('id')

        if not tool_name or not tool_call_id:
            logger.warning(f"[{name} - ID: {state.get('research_id')}] Skipping invalid tool call in message: {tool_call}")
            continue

        # Handle the FINISH signal - it shouldn't be executed as a tool here
        if isinstance(tool_name, str) and tool_name.upper() == "FINISH":
            logger.warning(f"[{name} - ID: {state.get('research_id')}] Agent attempted to call FINISH as a tool. Skipping tool execution for FINISH call.")
            continue

        logger.info(f"[{name} - ID: {state.get('research_id')}] Preparing to execute tool: {tool_name} (Call ID: {tool_call_id}) with input: {tool_input}")

        tool_invocation = ToolInvocation(tool=tool_name, tool_input=tool_input)
        tool_invocation_list.append(tool_invocation)

        try:
            response = tool_executor.invoke(tool_invocation)
            logger.info(f"[{name} - ID: {state.get('research_id')}] Tool '{tool_name}' execution completed. Response type: {type(response)}")

            # Standardize observation format (prefer string for LLM consumption)
            if isinstance(response, (str, dict, list)):
                 observation = json.dumps(response) if isinstance(response, (dict, list)) else response
            elif isinstance(response, ToolMessage):
                 observation = response.content
            else:
                 observation = str(response)

            intermediate_steps_updates.append((tool_invocation, observation))
            tool_messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))

        except Exception as e:
            logger.error(f"[{name} - ID: {state.get('research_id')}] Error executing tool {tool_name} (Call ID: {tool_call_id}): {e}", exc_info=True)
            error_message = f"Error executing tool {tool_name}: {str(e)}"
            intermediate_steps_updates.append((tool_invocation, error_message))
            tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call_id))

    new_intermediate_steps = state.get("intermediate_steps", []) + intermediate_steps_updates

    return {"messages": tool_messages, "intermediate_steps": new_intermediate_steps}


# --- Node for Final Report Generation ---
def generate_final_report_node(state: AgentState, name: str) -> Dict[str, Optional[ResearchReport]]:
    """Generates the final structured research report by calling the synthesis LLM."""
    research_id = state.get('research_id')
    logger.info(f"[{name} - ID: {research_id}] Generating final research report.")

    synthesis_model = get_primary_llm()
    if not synthesis_model:
        logger.error(f"[{name} - ID: {research_id}] Primary LLM (for synthesis) not configured. Cannot generate final report.")
        return {"final_result": None}

    intermediate_steps = state.get("intermediate_steps", [])
    original_query = state.get("query", "No query provided")

    max_evidence_steps = int(os.getenv("MAX_EVIDENCE_STEPS", "80"))
    max_observation_chars = int(os.getenv("MAX_OBSERVATION_CHARS", "600"))

    def _is_error_observation(observation_text: str) -> bool:
        lowered = observation_text.lower()
        return "error" in lowered or "exception" in lowered or "traceback" in lowered

    selected_steps = intermediate_steps
    if len(intermediate_steps) > max_evidence_steps:
        non_error_steps = [step for step in intermediate_steps if not _is_error_observation(str(step[1]))]
        if len(non_error_steps) >= max_evidence_steps:
            selected_steps = non_error_steps[-max_evidence_steps:]
        else:
            selected_steps = (non_error_steps + intermediate_steps)[-max_evidence_steps:]
        logger.warning(
            f"[{name} - ID: {research_id}] Trimming evidence steps from {len(intermediate_steps)} to "
            f"{len(selected_steps)} to reduce prompt size."
        )

    # --- Source Extraction and Filtering ---
    formatted_evidence = ""
    potential_sources: List[Source] = []
    unique_source_urls = set()

    logger.info(f"[{name} - ID: {research_id}] Formatting evidence and extracting sources from {len(selected_steps)} intermediate steps.")
    if len(selected_steps) < len(intermediate_steps):
        formatted_evidence += (
            f"\n[Note] Evidence trimmed: {len(intermediate_steps) - len(selected_steps)} steps omitted "
            "to stay within LLM limits.\n"
        )

    for step_index, (tool_invocation, observation) in enumerate(selected_steps):
        step_number = step_index + 1
        tool_name = tool_invocation.tool
        tool_input = tool_invocation.tool_input
        observation_text = str(observation)
        display_observation = observation_text[:max_observation_chars]
        if len(observation_text) > max_observation_chars:
            display_observation += "..."
        formatted_evidence += f"\n--- Step {step_number}: Tool Used: {tool_name} ---\n"
        formatted_evidence += f"Input: {str(tool_input)[:200]}{'...' if len(str(tool_input)) > 200 else ''}\n"
        formatted_evidence += f"Observation:\n{display_observation}\n"

        # Extract potential sources
        try:
            obs_data = json.loads(observation_text) if isinstance(observation, str) else observation
            if isinstance(obs_data, dict):
                # Handle various tool outputs
                results = obs_data.get('results') or obs_data.get('articles')
                if isinstance(results, list):
                    for item in results:
                        if isinstance(item, dict):
                            url = item.get('url') or item.get('link')
                            title = item.get('title')
                            snippet = item.get('snippet') or item.get('content') or item.get('description')
                            if url:
                                potential_sources.append(
                                    Source(url=url, title=title, snippet=snippet, tool_used=tool_name)
                                )

                elif tool_name == 'gemini_google_search_tool' and 'sources' in obs_data:
                    urls = obs_data.get('sources', [])
                    gemini_snippet = obs_data.get('summary') or (obs_data.get('key_facts')[0] if obs_data.get('key_facts') else 'N/A')
                    for url in urls:
                        if url:
                             potential_sources.append(
                                 Source(url=url, title=f"Source from Gemini Search for '{str(tool_input)[:50]}...'", snippet=gemini_snippet[:200], tool_used=tool_name)
                             )

                elif tool_name == 'firecrawl_scrape_tool' and 'url' in obs_data and 'markdown_content' in obs_data:
                    url = obs_data.get('url')
                    if url:
                        title = obs_data.get('metadata', {}).get('title') or url
                        snippet = (obs_data.get('markdown_content') or '')[:200] + '...'
                        potential_sources.append(
                            Source(url=url, title=title, snippet=snippet, tool_used=tool_name)
                        )

                elif tool_name == 'duckduckgo_search':
                     if isinstance(obs_data, list):
                         for item in obs_data:
                             if isinstance(item, dict) and 'url' in item and 'title' in item:
                                 url = item.get('url')
                                 title = item.get('title')
                                 snippet = item.get('snippet') or item.get('body')
                                 if url:
                                     potential_sources.append(
                                         Source(url=url, title=title, snippet=snippet, tool_used=tool_name)
                                     )
                     else:
                          logger.warning(f"[{name} - ID: {research_id}] Unexpected format for DuckDuckGo output in step {step_number}: {type(obs_data)}")

        except json.JSONDecodeError:
            logger.warning(f"[{name} - ID: {research_id}] Observation for step {step_number} is not valid JSON. Skipping source extraction for this step. Content: {str(observation)[:100]}...")
        except Exception as e:
            logger.warning(f"[{name} - ID: {research_id}] Error processing sources from step {step_number} observation: {e}", exc_info=False)

    # --- Filter and Deduplicate Sources ---
    filtered_sources: List[Source] = []
    for source in potential_sources:
        url_text = str(source.url) if source.url else ""
        # Check if URL is valid and not already added
        if (
            url_text and
            url_text.strip() and
            url_text.strip().lower() != "n/a" and
            url_text not in unique_source_urls
        ):
            if url_text.strip().lower().startswith(("http://", "https://")):
                filtered_sources.append(source)
                unique_source_urls.add(url_text)
            else:
                logger.warning(f"[{name} - ID: {research_id}] Filtering out source with invalid URL format: {url_text}")
        elif url_text in unique_source_urls:
            logger.debug(f"[{name} - ID: {research_id}] Skipping duplicate source URL: {url_text}")
        else:
            logger.warning(f"[{name} - ID: {research_id}] Filtering out source with invalid/missing URL: {url_text}")

    logger.info(f"[{name} - ID: {research_id}] Total unique and valid sources filtered for report: {len(filtered_sources)}")
    # --- End: Source Extraction and Filtering ---

    formatted_evidence += "\n-- End of Evidence --\n"

    # Prepare the prompt for the synthesis LLM
    # Pass the filtered sources to the parser
    prompt_context = {
        "query": original_query,
        "formatted_evidence": formatted_evidence,
        "sources": filtered_sources
    }
    synthesis_prompt_template = MARKET_REPORT_SYNTHESIS_TEMPLATE if state.get("market_research") else REPORT_SYNTHESIS_TEMPLATE
    synthesis_chain = synthesis_prompt_template | synthesis_model | report_parser

    logger.info(f"[{name} - ID: {research_id}] Invoking synthesis LLM chain...")
    try:
        final_report_object: ResearchReport = synthesis_chain.invoke(prompt_context)
        logger.info(f"[{name} - ID: {research_id}] Synthesis and parsing successful.")
        # Postprocess the generated report to improve structure, tone, and remove noise
        try:
            postprocessed = improve_report(final_report_object)
            if isinstance(postprocessed, ResearchReport):
                final_report_object = postprocessed
                logger.info(f"[{name} - ID: {research_id}] Postprocessing applied to final report.")
            else:
                logger.warning(f"[{name} - ID: {research_id}] Postprocessor returned error: %s", getattr(postprocessed, 'error', 'unknown'))
        except Exception as e:
            logger.warning(f"[{name} - ID: {research_id}] Error running postprocessor: {e}")
        # Ensure the filtered sources are part of the final report object
        if not final_report_object.sources:
             logger.warning(f"[{name} - ID: {research_id}] Final report object created but sources list is empty. Assigning filtered sources.")
             final_report_object.sources = filtered_sources
        else:
             merged_sources: List[Source] = []
             seen_urls = set()
             for source in final_report_object.sources + filtered_sources:
                 url_text = str(source.url) if source.url else ""
                 if not url_text or url_text in seen_urls:
                     continue
                 seen_urls.add(url_text)
                 merged_sources.append(source)
             if len(merged_sources) != len(final_report_object.sources):
                 logger.warning(
                     f"[{name} - ID: {research_id}] Expanded sources list from {len(final_report_object.sources)} to {len(merged_sources)} using filtered sources."
                 )
             final_report_object.sources = merged_sources

        return {"final_result": final_report_object}

    except OutputParserException as ope:
        logger.error(f"[{name} - ID: {research_id}] Failed to parse report: {ope}. Raw output was:\n{{Raw output omitted}}")
        return {"final_result": None}
    except Exception as e:
        logger.error(f"[{name} - ID: {research_id}] An unexpected error occurred during final report synthesis: {e}", exc_info=True)
        return {"final_result": None}


# --- Routing Logic ---
def should_continue(state: AgentState) -> str:
    """Determines whether to continue research, finish, or handle error."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    research_id = state.get('research_id')
    max_agent_steps = int(os.getenv("MAX_AGENT_STEPS", "24"))
    current_steps = len(state.get("intermediate_steps", []))

    if current_steps >= max_agent_steps:
        logger.warning(
            f"[Router - ID: {research_id}] Max agent steps reached ({current_steps}/{max_agent_steps}). "
            "Routing to generate_final_report."
        )
        return "generate_final_report"

    if not last_message:
        logger.error(f"[Router - ID: {research_id}] No messages in state. Ending run.")
        return END

    finish_called = False
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            for tc in last_message.tool_calls:
                name = tc.get("name")
                if isinstance(name, str) and name.upper() == "FINISH":
                    finish_called = True
                    logger.info(f"[Router - ID: {research_id}] FINISH tool call detected.")
                    break
        content = getattr(last_message, "content", None)
        content_text = content if isinstance(content, str) else ""
        if not finish_called and content_text and "FINISH" in content_text.upper():
            finish_called = True
            logger.info(f"[Router - ID: {research_id}] FINISH signal detected in message content.")

    if last_message.tool_calls and not finish_called:
        for tc in last_message.tool_calls:
            for v in tc.get('args', {}).values():
                if isinstance(v, str) and v.strip().upper() == "FINISH":
                    finish_called = True
                    logger.info(f"[Router - ID: {research_id}] FINISH signal detected in tool args.")
                    break
            if finish_called:
                break

    if finish_called:
        logger.info(f"[Router - ID: {research_id}] Routing to generate_final_report.")
        return "generate_final_report"
    elif isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info(f"[Router - ID: {research_id}] Routing to tool_node.")
        return "tool_node"
    elif isinstance(last_message, ToolMessage) or (isinstance(last_message, AIMessage) and not last_message.tool_calls):
         logger.info(f"[Router - ID: {research_id}] Routing back to agent_node.")
         return "agent_node"
    else:
        logger.warning(f"[Router - ID: {research_id}] Unhandled message type or state for routing: {type(last_message)}. Ending run.")
        return END


# --- Graph Construction ---
def create_web_research_agent_graph(config: Optional[Dict] = None):
    """Creates and compiles the LangGraph for the Web Research Agent."""
    logger.info("Creating Web Research Agent graph...")
    if config is None:
        config = {}

    streaming = False
    if isinstance(config, dict):
        streaming = bool(config.get("streaming", False))
    primary_llm = get_primary_llm(streaming=streaming)
    if not primary_llm:
        raise ValueError("Primary LLM could not be configured.")

    tools = create_agent_tools(config)
    llm_with_tools = primary_llm.bind_tools(tools)
    tool_executor = ToolExecutor(tools)

    agent_node_partial = lambda state: agent_node(state, agent=llm_with_tools, tools=tools, name="Agent")
    tool_node_partial = lambda state: tool_node(state, tool_executor=tool_executor, name="Action")
    generate_final_report_node_partial = lambda state: generate_final_report_node(state, name="SynthesizeReport")

    workflow = StateGraph(AgentState)

    workflow.add_node("agent_node", agent_node_partial)
    workflow.add_node("tool_node", tool_node_partial)
    workflow.add_node("generate_final_report", generate_final_report_node_partial)

    workflow.set_entry_point("agent_node")

    workflow.add_conditional_edges(
        "agent_node",
        should_continue,
        {
            "tool_node": "tool_node",
            "generate_final_report": "generate_final_report",
            "agent_node": "agent_node",
            END: END,
        },
    )

    workflow.add_edge("tool_node", "agent_node")
    workflow.add_edge("generate_final_report", END)

    app = workflow.compile()
    logger.info("Web Research Agent graph created and compiled.")
    return app


# --- Main Execution Logic ---
async def run_web_research(query: str, config: Optional[Dict] = None):
    """Runs the web research agent for a given query."""
    research_id = str(uuid.uuid4())
    logger.info(f"Starting web research for query: '{query}' (ID: {research_id})")
    if config is None:
        config = {}

    app = create_web_research_agent_graph(config)

    market_research = bool(config.get("market_research")) if isinstance(config, dict) else False

    initial_state = AgentState(
        query=query,
        research_id=research_id,
        market_research=market_research,
        messages=[HumanMessage(content=query)],
        intermediate_steps=[],
        final_result=None
    )

    try:
        recursion_limit = int(os.getenv("RECURSION_LIMIT", "80"))
        if isinstance(config, dict) and "recursion_limit" in config:
            recursion_limit = int(config["recursion_limit"])
        final_state = app.invoke(initial_state, config={"recursion_limit": recursion_limit})

        logger.info(f"Research completed for ID: {research_id}")
        final_result = final_state.get('final_result')

        if isinstance(final_result, ResearchReport):
            logger.info(f"Final report generated for ID: {research_id}. Summary: {final_result.summary[:100]}...")
            return final_result
        else:
            logger.error(f"Research finished for ID: {research_id}, but no valid report was generated in the final state.")
            return ErrorResponse(error="Failed to generate report", details="The agent finished, but the final report was not found or invalid.")

    except Exception as e:
        logger.error(f"Error during research execution for ID: {research_id}: {e}", exc_info=True)
        return ErrorResponse(error="Agent execution failed", details=str(e))


