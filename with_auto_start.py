import os
import base64
import uuid
from datetime import datetime, timedelta
from typing import Annotated, Literal, Optional
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import requests
import json
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
import nest_asyncio
from langchain_groq import ChatGroq

load_dotenv()

# Define the State schema
def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[list[Literal["scrumagent", "azure_agent", "deadline_agent", "weekly_agent"]], update_dialog_stack]

# Utility: Handle tool errors
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")


# Entry node creator
def create_entry_node(assistant_name: str, new_dialog_state: str):
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"] if state["messages"][-1].tool_calls else str(uuid.uuid4())
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation and use the provided tools to assist the user. "
                            f"Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": [new_dialog_state]
        }
    return entry_node

# Print event utility
def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


# Tool: Fetch Azure Board Data
class AzureQuery(BaseModel):
    select: list[str] = Field(
        default=["System.Id", "System.Title", "System.Description", "System.AssignedTo", "Microsoft.VSTS.Common.AcceptanceCriteria", "System.State"],
        description="List of fields to select, e.g., ['System.Id', 'System.Title']"
    )
    where: dict[str, str] = Field(default={}, description="Filters as field-value pairs, e.g., {'System.State': 'Active'}")
    order_by: Optional[str] = Field(default=None, description="Field to order by, e.g., 'System.CreatedDate'")
    limit: Optional[int] = Field(default=None, description="Maximum number of items to return, e.g., 10")
    check_missing_in_new: bool = Field(default=False, description="If true, check for missing values in 'New' user stories")
class CreateWorkItemInput(BaseModel):
    work_item_type: str = Field(description="Type of the work item, e.g., 'User Story', 'Task', 'Bug'")
    title: str = Field(description="Title of the work item")
    description: Optional[str] = Field(default=None, description="Description of the work item")
    assigned_to: Optional[str] = Field(default=None, description="User to assign the work item to")

class UpdateWorkItemInput(BaseModel):
    work_item_id: int = Field(description="ID of the work item to update")
    fields: dict[str, str] = Field(description="Fields to update, e.g., {'System.Title': 'New Title', 'System.State': 'Closed'}")

class DeleteWorkItemInput(BaseModel):
    work_item_id: int = Field(description="ID of the work item to delete")

def get_azure_api_config():
    organization = os.getenv("ORGANIZATION")
    project = os.getenv("PROJECT")
    api_token = os.getenv("AZURE_API_TOKEN")
    if not all([organization, project, api_token]):
        raise ValueError("Missing Azure configuration (organization, project, or API token).")
    base_url = f"https://dev.azure.com/{organization}/{project}/_apis/"
    token_bytes = f":{api_token}".encode("utf-8")
    base64_token = base64.b64encode(token_bytes).decode("utf-8")
    headers = {
        "Authorization": f"Basic {base64_token}",
        "Content-Type": "application/json"
    }
    return base_url, headers

import html2text
import json

def clean_html(text):
    if not text:
        return ""
    h = html2text.HTML2Text()
    h.ignore_links = True
    return h.handle(text).strip()

@tool
def fetch_azure_board_data(query: AzureQuery) -> str:
    """Fetches work items from Azure Boards based on the provided query."""
    logger.info(f"Invoking fetch_azure_board_data with query: {query}")
    base_url, headers = get_azure_api_config()

    wiql = "SELECT [System.Id] FROM WorkItems"
    if query.check_missing_in_new:
        wiql += " WHERE [System.WorkItemType] = 'User Story' AND [System.State] = 'New'"
    elif query.where:
        where_clauses = [f"[{k}] = '{v}'" for k, v in query.where.items()]
        wiql += " WHERE " + " AND ".join(where_clauses)
    if query.order_by and not query.check_missing_in_new:
        wiql += f" ORDER BY [{query.order_by}]"

    wiql_query = {"query": wiql}
    wiql_url = base_url + "wit/wiql?api-version=7.2-preview.2"

    try:
        response = requests.post(wiql_url, headers=headers, json=wiql_query)
        response.raise_for_status()
        result = response.json()
        work_item_ids = [item["id"] for item in result.get("workItems", [])]

        if not work_item_ids:
            return "No work items found matching the criteria."

        if query.limit and not query.check_missing_in_new:
            work_item_ids = work_item_ids[:query.limit]

        fields = query.select
        ids_str = ",".join(map(str, work_item_ids))
        work_items_url = base_url + f"wit/workitems?ids={ids_str}&fields={','.join(fields)}&api-version=7.2-preview.3"
        detail_response = requests.get(work_items_url, headers=headers)
        detail_response.raise_for_status()
        work_items = detail_response.json()["value"]
        logger.info(f"Raw work items response: {json.dumps(work_items, indent=2)}")

        if query.check_missing_in_new:
            missing_info = []
            for item in work_items:
                fields = item.get("fields", {})
                missing_fields = []
                if not fields.get("System.Title"):
                    missing_fields.append("Title")
                if not clean_html(fields.get("System.Description")):
                    missing_fields.append("Description")
                if not fields.get("System.AssignedTo", {}).get("displayName"):
                    missing_fields.append("AssignedTo")
                if not clean_html(fields.get("Microsoft.VSTS.Common.AcceptanceCriteria")):
                    missing_fields.append("AcceptanceCriteria")
                if missing_fields:
                    missing_info.append(f"User Story {fields.get('System.Id')}: Missing {', '.join(missing_fields)}")
            if missing_info:
                return "User stories in 'New' state with missing values:\n" + "\n".join(missing_info)
            return "No missing values found in 'New' user stories."
        else:
            formatted_items = []
            for item in work_items:
                fields = item.get("fields", {})
                item_info = f"User Story {fields.get('System.Id')}:\n"
                item_info += f"  Title: {fields.get('System.Title', 'N/A')}\n"
                item_info += f"  Description: {clean_html(fields.get('System.Description', 'N/A'))}\n"
                item_info += f"  Assigned To: {fields.get('System.AssignedTo', {}).get('displayName', 'N/A')}\n"
                item_info += f"  Acceptance Criteria: {clean_html(fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', 'N/A'))}\n"
                item_info += f"  State: {fields.get('System.State', 'N/A')}"
                formatted_items.append(item_info)
            return "\n\n".join(formatted_items) if formatted_items else "No data to display."
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Azure data: {str(e)}")
        return f"Error fetching Azure data: {str(e)}"



WORK_ITEM_TYPE_MAPPING = {
    "user story": "User Story",
    "task": "Task",
    "bug": "Bug"
}
from urllib.parse import quote
import requests

@tool
def create_azure_work_item(input: CreateWorkItemInput) -> str:
    """Creates a new work item in Azure Boards."""
    base_url, headers = get_azure_api_config()
    headers["Content-Type"] = "application/json-patch+json"  # Correct Content-Type
    work_item_type = WORK_ITEM_TYPE_MAPPING.get(input.work_item_type.lower(), input.work_item_type)
    work_item_type_encoded = quote(work_item_type)
    url = base_url + f"wit/workitems/${work_item_type_encoded}?api-version=7.2-preview.3"
    body = [
        {"op": "add", "path": "/fields/System.Title", "value": input.title}
    ]
    if input.description:
        body.append({"op": "add", "path": "/fields/System.Description", "value": input.description})
    if input.assigned_to:
        body.append({"op": "add", "path": "/fields/System.AssignedTo", "value": input.assigned_to})
    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        created_item = response.json()
        return f"Work item created successfully with ID: {created_item['id']}"
    except requests.exceptions.RequestException as e:
        return f"Error creating work item: {str(e)}"
@tool
def update_azure_work_item(input: UpdateWorkItemInput) -> str:
    """Updates an existing work item in Azure Boards."""
    base_url, headers = get_azure_api_config()
    headers["Content-Type"] = "application/json-patch+json"  # Correct Content-Type
    url = base_url + f"wit/workitems/{input.work_item_id}?api-version=7.2-preview.3"
    body = [
        {"op": "add", "path": f"/fields/{field}", "value": value}
        for field, value in input.fields.items()
    ]
    try:
        response = requests.patch(url, headers=headers, json=body)
        response.raise_for_status()
        return f"Work item {input.work_item_id} updated successfully."
    except requests.exceptions.RequestException as e:
        return f"Error updating work item: {str(e)}"
@tool
def delete_azure_work_item(input: DeleteWorkItemInput) -> str:
    """Deletes a work item from Azure Boards."""
    base_url, headers = get_azure_api_config()
    url = base_url + f"wit/workitems/{input.work_item_id}?api-version=7.2-preview.3"
    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        return f"Work item {input.work_item_id} deleted successfully."
    except requests.exceptions.RequestException as e:
        return f"Error deleting work item: {str(e)}"

# Manually Defined Deadlines
MANUAL_DEADLINES = {
    "151": "2025-04-05",
    "131": "2025-03-30",
    "146": "2025-04-10",
    "all_user_stories": "2025-04-07"
}


# Tool: Check Manual Deadlines
class DeadlineQuery(BaseModel):
    work_item_id: Optional[str] = Field(default=None, description="Specific work item ID to check deadline for, e.g., '12345'")
    category: Optional[str] = Field(default=None, description="Category like 'all_user_stories' to check a group deadline")
    check_all: bool = Field(default=False, description="If true, check all manually defined deadlines and return those within the threshold")

@tool
def check_manual_deadlines(query: DeadlineQuery) -> str:
    """Checks manually defined deadlines and compares them to the current date."""
    logger.info(f"Invoking check_manual_deadlines with query: {query}")
    current_date = datetime.now()  # Use real current date
    threshold_days = 3

    if query.check_all:
        approaching = []
        for key, deadline_str in MANUAL_DEADLINES.items():
            try:
                deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d")
                time_diff = (deadline_date - current_date).days
                if time_diff <= threshold_days:
                    status = f"past due by {-time_diff} days" if time_diff < 0 else f"{time_diff} day{'s' if time_diff > 1 else ''} left"
                    approaching.append(f"{key}: {deadline_str} ({status})")
            except ValueError:
                continue
        if approaching:
            return "Approaching deadlines:\n" + "\n".join(approaching)
        else:
            return "No approaching deadlines within the next 3 days."
    elif query.work_item_id:
        deadline_str = MANUAL_DEADLINES.get(query.work_item_id)
        if not deadline_str:
            return f"No deadline found for work item {query.work_item_id}."
    elif query.category:
        deadline_str = MANUAL_DEADLINES.get(query.category)
        if not deadline_str:
            return f"No deadline found for category {query.category}."
    else:
        return "Please specify a work item ID, category, or set check_all to True."

    try:
        deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d")
        time_diff = (deadline_date - current_date).days
        if time_diff < 0:
            return f"The deadline for {query.work_item_id or query.category} was {deadline_str}. It's past due by {-time_diff} days."
        elif time_diff == 0:
            return f"The deadline for {query.work_item_id or query.category} is today ({deadline_str})!"
        elif time_diff <= threshold_days:
            return f"The deadline for {query.work_item_id or query.category} is {deadline_str}. Only {time_diff} day{'s' if time_diff > 1 else ''} left!"
        else:
            return f"The deadline for {query.work_item_id or query.category} is {deadline_str}. {time_diff} days remaining."
    except ValueError:
        return f"Invalid deadline format for {query.work_item_id or query.category}: {deadline_str}"

# Tool: Weekly Status Report
class WeeklyQuery(BaseModel):
    period: str = Field(default="current", description="Time period: 'current' for this week, 'next' for next week")
    team: Optional[str] = Field(default=None, description="Specific team name if applicable")


@tool
def generate_weekly_status(query: WeeklyQuery) -> str:
    """Generates a weekly status report based on the specified period."""
    current_date = datetime(2025, 4, 3)  # Using current date from your setup

    if query.period == "current":
        week_start = current_date - timedelta(days=current_date.weekday())
        week_end = week_start + timedelta(days=6)
        period_desc = "this week"
    elif query.period == "next":
        week_start = current_date - timedelta(days=current_date.weekday()) + timedelta(days=7)
        week_end = week_start + timedelta(days=6)
        period_desc = "next week"
    else:
        return "Error: Invalid period. Use 'current' or 'next'."

    team_info = f" for team {query.team}" if query.team else ""
    return f"Weekly Status Report{team_info} ({period_desc}):\n" \
           f"Period: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}\n" \
           f"- Sprint Progress: 75% complete\n" \
           f"- Blockers: 2 issues pending resolution\n" \
           f"- Planned Tasks: 15 user stories scheduled\n" \
           f"(Note: This is a sample report. In a real implementation, this would fetch actual data.)"


# Scrum Master Agent
class ScrumMasterAgent:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        result = self.runnable.invoke(state)
        # Only re-invoke if result is empty or invalid
        if not result.tool_calls and (
            not result.content or
            (isinstance(result.content, str) and result.content.startswith("<tool-use>"))
        ):
            messages = state["messages"] + [HumanMessage(content="Respond with a real output.")]
            state = {**state, "messages": messages}
            result = self.runnable.invoke(state)
        return {"messages": result}

# Control Tools
class CompleteOrEscalate(BaseModel):
    """Mark the task as completed or escalate to the main assistant."""
    cancel: bool = True
    reason: str


class ToAzureAssisstant(BaseModel):
    """Delegate to the Azure Assistant."""
    request: str = Field(description="Fetch all the data and answer the user's question.")


class ToDeadlineAssistant(BaseModel):
    """Delegate to the Deadline Assistant."""
    request: str = Field(description="Check deadline data and answer the user's question.")


class ToWeeklyAssistant(BaseModel):
    """Delegate to the Weekly Assistant."""
    request: str = Field(description="Generate weekly status or planning data and answer the user's question.")


# Azure Info Agent Node
from groq import BadRequestError

def azureinfo_agent_node(state: State, config: RunnableConfig):
    try:
        logger.info(f"Azure agent processing state: {state}")
        result = azureinfo_runnable.invoke(state)
        logger.info(f"Azure agent LLM output: {result}")
        if not result.tool_calls and (
            not result.content or
            isinstance(result.content, str) and result.content.startswith("<tool-use>")
        ):
            messages = state["messages"] + [HumanMessage(content="Please use the fetch_azure_board_data tool to check for missing values or retrieve data as requested.")]
            state = {**state, "messages": messages}
            result = azureinfo_runnable.invoke(state)
            logger.info(f"Azure agent re-invoked with fallback: {result}")
        return {"messages": result}
    except Exception as e:
        logger.error(f"Error in azureinfo_agent_node: {str(e)}")
        error_message = f"Error processing Azure request: {str(e)}. Please clarify your request or try again."
        return {
            "messages": [
                AIMessage(content=error_message)
            ]
        }

# Deadline Agent Node
def deadline_agent_node(state: State, config: RunnableConfig):
    if not state.get("dialog_state"):
        state["dialog_state"] = ["deadline_agent"]
    result = deadline_runnable.invoke(state)
    if not result.tool_calls and (
        not result.content or
        isinstance(result.content, str) and result.content.startswith("<tool-use>")
    ):
        # Re-invoke with a more specific prompt
        messages = state["messages"] + [HumanMessage(content="Please use the check_manual_deadlines tool to retrieve deadline information as requested.")]
        state = {**state, "messages": messages}
        result = deadline_runnable.invoke(state)
    return {"messages": result}

# Weekly Agent Node
def weekly_agent_node(state: State, config: RunnableConfig):
    if not state.get("dialog_state"):
        state["dialog_state"] = ["weekly_agent"]
    result = weekly_runnable.invoke(state)
    if not result.tool_calls and (
            not result.content or isinstance(result.content, list) and not result.content[0].get("text")
    ):
        messages = state["messages"] + [("user", "Respond with a real output.")]
        state = {**state, "messages": messages}
        result = weekly_runnable.invoke(state)
    return {"messages": result}


# Instantiate LLM and Runnables
# llm = ChatOllama(base_url="http://192.168.5.58:11434", model="qwen2.5:32b")
# llm = ChatOllama(base_url="http://192.168.0.164:11436", model="qwen2.5:32b")
# llm = ChatOllama(base_url="http://192.168.0.7:11461", model="qwen2.5:32b")
# llm = ChatGroq(model="llama3-70b-8192",api_key=os.getenv("groq_api"))
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.2, api_key=os.getenv("GEMINI_API_KEY"))

scrum_master_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sarcastic multi-language Scrum Master Agent. Greet everyone initially. Greet them with different reply everytime."
               "Reply for evbest company to apply for agentic ai fresher jobsery query in a very polite way and crisp. "
               "If the user asks about Azure Boards, mentions user story, "
               "or requests 'missing values in new user stories', immediately delegate to the Azure agent. "
               "If the user asks about deadlines or due dates, delegate to the Deadline agent. "
               "If the user asks about weekly status, planning, or sprint progress, delegate to the Weekly agent."),
    ("placeholder", "{messages}")
])
scrum_master_runnable = scrum_master_prompt | llm.bind_tools(
    [ToAzureAssisstant, ToDeadlineAssistant, ToWeeklyAssistant],
    tool_choice="auto"  # Prevent automatic tool calls unless explicitly triggered
)
from langchain_core.prompts import ChatPromptTemplate
azureinfo_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AzureInfo agent responsible for performing CRUD operations on Azure Boards work items. "
     "You MUST use the provided tools to fetch or manipulate data and MUST NOT generate fabricated responses. "
     "Based on the user's request, use the appropriate tool:\n"
     "- For checking missing values in 'New' user stories (e.g., 'check for missing values in new user stories'), call fetch_azure_board_data with check_missing_in_new=True.\n"
     "- For reading data (e.g., 'show me all the user stories with state active'), call fetch_azure_board_data with the default select fields: ['System.Id', 'System.Title', 'System.Description', 'System.AssignedTo', 'Microsoft.VSTS.Common.AcceptanceCriteria', 'System.State'], unless the user explicitly specifies other fields.\n"
     "- For creating a work item, call create_azure_work_item.\n"
     "- For updating a work item, call update_azure_work_item.\n"
     "- For deleting a work item, call delete_azure_work_item.\n"
     "If the request is unclear or missing information, ask for clarification and do not proceed without a tool call. "
     "Return the tool's output directly without adding extra text unless clarification is needed. "
     "DO NOT fabricate data like user story IDs or statuses. Always rely on the tool's response."),
    ("placeholder", "{messages}")
])
azureinfo_runnable = azureinfo_prompt | llm.bind_tools(
    [fetch_azure_board_data, create_azure_work_item, update_azure_work_item, delete_azure_work_item, CompleteOrEscalate],
    tool_choice="auto"
)

deadline_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Deadline agent responsible for checking deadlines. "
     "You MUST use the check_manual_deadlines tool for all deadline-related queries and MUST NOT generate fabricated responses. "
     "Construct the DeadlineQuery based on the user's request:\n"
     "- For 'check for approaching deadlines', call check_manual_deadlines with check_all=True.\n"
     "- For 'what’s the deadline for work item 12345', call check_manual_deadlines with work_item_id='12345'.\n"
     "- For 'show deadlines for all user stories', call check_manual_deadlines with category='all_user_stories'.\n"
     "If the query is incomplete, ask for clarification and do not proceed without a tool call. "
     "Return the tool's output directly without adding extra text unless clarification is needed. "
     "DO NOT fabricate deadlines or task names. Always rely on the tool's response."),
    ("placeholder", "{messages}")
])
deadline_runnable = deadline_prompt | llm.bind_tools(
    [check_manual_deadlines, CompleteOrEscalate],
    tool_choice="auto"
)
weekly_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Weekly agent. When a user asks about weekly status, planning, or sprint progress, "
               "use the generate_weekly_status tool. Construct the WeeklyQuery based on the user's request:\n"
               "- For 'what’s the status for this week', use period='current'.\n"
               "- For 'show next week’s plan', use period='next'.\n"
               "- For 'weekly status for Team A', use period='current', team='Team A'.\n"
               "If the query is unclear, ask for clarification. The tool returns a status report string; present it directly to the user."),
    ("placeholder", "{messages}")
])
weekly_runnable = weekly_prompt | llm.bind_tools([generate_weekly_status, CompleteOrEscalate])

# Define the Graph
thread_id = str(uuid.uuid4())
config = {"configurable": {"user_id": "3442 587242", "thread_id": thread_id}}
builder = StateGraph(State)

# Add nodes
builder.add_node("scrum_master_agent", ScrumMasterAgent(scrum_master_runnable))
builder.add_node("enter_azure_agent", create_entry_node("Azure Assistant", "azure_agent"))
builder.add_node("azureinfo_agent", azureinfo_agent_node)
azure_tools = [fetch_azure_board_data, create_azure_work_item, update_azure_work_item, delete_azure_work_item]
builder.add_node("azure_tools", create_tool_node_with_fallback(azure_tools))
builder.add_node("enter_deadline_agent", create_entry_node("Deadline Assistant", "deadline_agent"))
builder.add_node("deadline_agent", deadline_agent_node)
builder.add_node("deadline_tools", create_tool_node_with_fallback([check_manual_deadlines]))
builder.add_node("enter_weekly_agent", create_entry_node("Weekly Assistant", "weekly_agent"))
builder.add_node("weekly_agent", weekly_agent_node)
builder.add_node("weekly_tools", create_tool_node_with_fallback([generate_weekly_status]))
# Routing Functions
def route_scrum_master(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        if tool_name == ToAzureAssisstant.__name__:
            return "to_azure_agent"
        elif tool_name == ToDeadlineAssistant.__name__:
            return "to_deadline_agent"
        elif tool_name == ToWeeklyAssistant.__name__:
            return "to_weekly_agent"
    # Fallback to continue if no tool calls
    return "continue"


def route_azureinfo(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if last_message.tool_calls[0]["name"] == "CompleteOrEscalate":
            return "escalate"
        return "azure_tools"
    return "continue"


def route_deadline(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if last_message.tool_calls[0]["name"] == "CompleteOrEscalate":
            return "escalate"
        return "deadline_tools"
    return "continue"


def route_weekly(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if last_message.tool_calls[0]["name"] == "CompleteOrEscalate":
            return "escalate"
        return "weekly_tools"
    return "continue"


# Add Edges
builder.add_edge(START, "scrum_master_agent")
builder.add_conditional_edges("scrum_master_agent", route_scrum_master, {
    "to_azure_agent": "enter_azure_agent",
    "to_deadline_agent": "enter_deadline_agent",
    "to_weekly_agent": "enter_weekly_agent",
    "continue": END
})
builder.add_edge("enter_azure_agent", "azureinfo_agent")
builder.add_conditional_edges("azureinfo_agent", route_azureinfo, {
    "azure_tools": "azure_tools",
    "escalate": "scrum_master_agent",
    "continue": END
})
builder.add_edge("azure_tools", "azureinfo_agent")
builder.add_edge("enter_deadline_agent", "deadline_agent")
builder.add_conditional_edges("deadline_agent", route_deadline, {
    "deadline_tools": "deadline_tools",
    "escalate": "scrum_master_agent",
    "continue": END
})
builder.add_edge("deadline_tools", "deadline_agent")
builder.add_edge("enter_weekly_agent", "weekly_agent")
builder.add_conditional_edges("weekly_agent", route_weekly, {
    "weekly_tools": "weekly_tools",
    "escalate": "scrum_master_agent",
    "continue": END
})
builder.add_edge("weekly_tools", "weekly_agent")


# Compile the Graph
memory = MemorySaver()
multi_agent_graph = builder.compile(checkpointer=memory)

import nest_asyncio
nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles


# Save the graph as a PNG file
png_file_path = "graph.png"
png_image = multi_agent_graph.get_graph().draw_mermaid_png(
    curve_style=CurveStyle.LINEAR,
    node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
    wrap_label_n_words=9,
    output_file_path=png_file_path,  # Specify the file path to save the PNG
    draw_method=MermaidDrawMethod.PYPPETEER,
    background_color="white",
    padding=10,
)
print(f"Graph saved as {png_file_path}")

_printed = set()

nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

load_dotenv()

# [Your existing State, Tools, and Agent definitions remain unchanged up to the graph compilation]

# Define the Graph (unchanged up to compilation)
thread_id = str(uuid.uuid4())
config = {"configurable": {"user_id": "3442 587242", "thread_id": thread_id}}
memory = MemorySaver()
multi_agent_graph = builder.compile(checkpointer=memory)

# Telegram Bot Setup
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


# Function to print event details (modified to use logger)
def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        logger.info(f"Currently in dialog state: {current_state[-1]}")
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr()
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            logger.info(f"Message: {msg_repr}")
            _printed.add(message.id)


# Function to process user input through the workflow with detailed logging
import os
import base64
import uuid
from datetime import datetime, timedelta
from typing import Annotated, Literal, Optional
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import json
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
import nest_asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import timezone

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# [Your existing State, Tools, Agent definitions, and Graph compilation remain unchanged]

# Telegram Bot Setup
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Function to print event details
def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        logger.info(f"Currently in dialog state: {current_state[-1]}")
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr()
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            logger.info(f"Message: {msg_repr}")
            _printed.add(message.id)

# Function to process user input through the workflow
async def process_message(message: str, user_id: str) -> str:
    logger.info(f"Processing message from user {user_id}: '{message}'")
    state = {
        "messages": [HumanMessage(content=message)],
        "user_info": f"User ID: {user_id}"
    }
    config["configurable"]["user_id"] = user_id

    final_response = ""
    _printed = set()
    for event in multi_agent_graph.stream(state, config, stream_mode="values"):
        _print_event(event, _printed)
        messages = event.get("messages", [])
        if messages and isinstance(messages, list):
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage) and last_message.content:
                final_response = last_message.content
            elif isinstance(last_message, AIMessage) and last_message.content and not last_message.content.startswith("<tool-use>"):
                final_response = last_message.content

    if not final_response:
        result = multi_agent_graph.invoke(state, config)
        for msg in reversed(result["messages"]):
            if isinstance(msg, ToolMessage) and msg.content:
                final_response = msg.content
                break
            elif isinstance(msg, AIMessage) and msg.content and not msg.content.startswith("<tool-use>"):
                final_response = msg.content
                break

    if not final_response:
        final_response = "Sorry, I couldn't process that request. Please try again or clarify your query."

    logger.info(f"Final response to user {user_id}: {final_response}")
    return final_response

# Telegram Handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.message.from_user.id)
    await update.message.reply_text("Hello! I'm your Scrum Master bot. Let me check for approaching deadlines and missing values in user stories with state 'new'.")

    missing_values_query = "Check for missing 'description', 'acceptance criteria' and 'assign To' in user stories with state 'new'"
    missing_values_response = await process_message(missing_values_query, user_id)
    await update.message.reply_text(missing_values_response)

    deadline_query = "Check for approaching deadlines"
    deadline_response = await process_message(deadline_query, user_id)
    await update.message.reply_text(deadline_response)

    await update.message.reply_text("Standing by. Tag @acidaes_bot with your request.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_message = update.message.text
        user_id = str(update.message.from_user.id)
        if update.message.chat.type in ["group", "supergroup"]:
            bot_username = context.bot.username
            if f"@{bot_username}" not in user_message:
                return
        response = await process_message(user_message, user_id)
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        await update.message.reply_text("An error occurred. Please try again later.")

# Main function to run the bot with scheduler
def main() -> None:
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN not found in .env file")

    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Set up scheduler
    tz = timezone('Asia/Kolkata')  # Indian Standard Time
    scheduler = AsyncIOScheduler(timezone=tz)

    async def daily_update():
        chat_id = "-4702403141"
        logger.info("Running daily update at 1:12 PM")

        # Process missing values query
        missing_values_query = "Check for missing 'description', 'acceptance criteria' and 'assign To' in user stories with state 'new'"
        missing_values_response = await process_message(missing_values_query, "automated_task")

        # Process deadlines query
        deadline_query = "Check for approaching deadlines"
        deadline_response = await process_message(deadline_query, "automated_task")

        # Combine and send the update
        message = "🌅 **Daily Update at 9:00 AM**:\n\n"
        message += "❌ **Missing Values in New User Stories**:\n" + missing_values_response + "\n\n"
        message += "-----------------------------------------\n\n"
        message += "⏰ **Approaching Deadlines**:\n" + deadline_response
        await application.bot.send_message(chat_id=chat_id, text=message)
        logger.info(f"Daily update sent to chat {chat_id}")

    # Schedule the job to run every day at 1:12 PM
    scheduler.add_job(daily_update, 'cron', hour=13, minute=32)
    scheduler.start()

    # Start the bot
    logger.info("Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
