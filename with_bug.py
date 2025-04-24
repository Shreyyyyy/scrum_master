import os
import base64
import uuid
import json
import threading
import logging
import nest_asyncio
import requests
import html2text
from datetime import datetime
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
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import timezone
from urllib.parse import quote

# Apply nest_asyncio for async compatibility
nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
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
    dialog_state: Annotated[
        list[Literal["scrumagent", "azure_agent", "deadline_agent", "weekly_agent"]], update_dialog_stack]
    deadline_results: Optional[str]  # Store deadline check results


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
        tool_call_id = state["messages"][-1].tool_calls[0]["id"] if state["messages"][-1].tool_calls else str(
            uuid.uuid4())
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


# Azure API Configuration
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


def clean_html(text):
    if not text:
        return ""
    h = html2text.HTML2Text()
    h.ignore_links = True
    return h.handle(text).strip()

# Azure Tools
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
    work_item_type: str = Field(description="Type of the work item")
    title: str = Field(description="Title of the work item")
    description: Optional[str] = Field(default=None, description="Description of the work item")
    assigned_to: Optional[str] = Field(default=None, description="User to assign the work item to")


class UpdateWorkItemInput(BaseModel):
    work_item_id: int = Field(description="ID of the work item to update")
    fields: dict[str, str] = Field(description="Fields to update")


class DeleteWorkItemInput(BaseModel):
    work_item_id: int = Field(description="ID of the work item to delete")


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
        wiql += " WHERE [System.TeamProject] = 'DevFusion2' AND [System.WorkItemType] = 'User Story' AND [System.State] = 'New'"
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

class QAFailStoriesQuery(BaseModel):
    """Query to fetch user stories in 'QA Fail' state with their child bugs."""
    pass  # No parameters needed for this query as per the tool implementation

class BugDetailsQuery(BaseModel):
    """Query to fetch details of a specific bug by ID or title."""
    bug_id: Optional[int] = Field(default=None, description="ID of the bug to fetch details for")
    title: Optional[str] = Field(default=None, description="Title of the bug to fetch details for")

@tool
def fetch_qa_fail_stories_with_bugs(query: QAFailStoriesQuery) -> str:
    """Fetches user stories in 'QA Fail' state with their child bugs and details."""
    base_url, headers = get_azure_api_config()
    wiql = "SELECT [System.Id] FROM WorkItems WHERE [System.WorkItemType] = 'User Story' AND [System.State] = 'QA Fail' AND [System.TeamProject] = 'DevFusion2'"
    wiql_query = {"query": wiql}
    wiql_url = base_url + "wit/wiql?api-version=7.2-preview.2"
    response = requests.post(wiql_url, headers=headers, json=wiql_query)
    response.raise_for_status()
    result = response.json()
    story_ids = [item["id"] for item in result.get("workItems", [])]
    if not story_ids:
        return "No user stories found in 'QA Fail' state."
    stories_with_bugs = []
    for story_id in story_ids:
        work_item_url = base_url + f"wit/workitems/{story_id}?$expand=relations&api-version=7.2-preview.3"
        work_item_response = requests.get(work_item_url, headers=headers)
        work_item_response.raise_for_status()
        work_item = work_item_response.json()
        child_bugs = []
        if "relations" in work_item:
            for relation in work_item["relations"]:
                if relation["rel"] == "System.LinkTypes.Hierarchy-Forward" and "workitems" in relation["url"]:
                    bug_id = int(relation["url"].split("/")[-1])
                    bug_url = base_url + f"wit/workitems/{bug_id}?fields=System.Id,System.Title,System.State,System.AssignedTo&api-version=7.2-preview.3"
                    bug_response = requests.get(bug_url, headers=headers)
                    bug_response.raise_for_status()
                    bug_data = bug_response.json()
                    bug_fields = bug_data["fields"]
                    assigned_to = bug_fields.get("System.AssignedTo", {}).get("displayName", "Unassigned")
                    mention = f"@{assigned_to}" if assigned_to != "Unassigned" else "Unassigned"
                    child_bugs.append({
                        "id": bug_fields.get("System.Id"),
                        "title": bug_fields.get("System.Title"),
                        "state": bug_fields.get("System.State"),
                        "assigned_to": mention
                    })
        stories_with_bugs.append({
            "story_id": work_item["id"],
            "story_title": work_item["fields"].get("System.Title"),
            "bugs": child_bugs
        })
    output = []
    for story in stories_with_bugs:
        output.append(f"User Story {story['story_id']}: {story['story_title']}")
        if story['bugs']:
            output.append("  Child Bugs:")
            for bug in story['bugs']:
                output.append(f"    - Bug {bug['id']}: {bug['title']} (State: {bug['state']}, Assigned to: {bug['assigned_to']})")
                if bug['state'] not in ["Closed", "Resolved"]:
                    output.append(f"      {bug['assigned_to']}, please complete this bug.")
        else:
            output.append("  No child bugs found.")
    return "\n".join(output) if output else "No data to display."

@tool
def get_bug_details(query: BugDetailsQuery) -> str:
    """
    Fetches detailed information about a specific bug from Azure Boards by ID or title.
    If both ID and title are provided, prioritizes ID. If only title is provided, fetches
    the first matching bug (case-insensitive). Returns an error if multiple bugs match the title.
    """
    logger.info(f"Invoking get_bug_details with query: {query}")
    base_url, headers = get_azure_api_config()
    fields = [
        "System.Id", "System.Title", "System.Description", "System.State",
        "System.AssignedTo", "Microsoft.VSTS.Common.Priority",
        "System.CreatedDate", "System.ChangedDate"
    ]

    try:
        if query.bug_id:
            # Fetch by ID
            bug_url = base_url + f"wit/workitems/{query.bug_id}?fields={','.join(fields)}&api-version=7.2-preview.3"
            response = requests.get(bug_url, headers=headers)
            response.raise_for_status()
            bug_data = response.json()
            fields_data = bug_data["fields"]
        elif query.title:
            # Fetch by title using WIQL
            wiql = (
                f"SELECT [System.Id] FROM WorkItems "
                f"WHERE [System.WorkItemType] = 'Bug' "
                f"AND [System.Title] = '{query.title}' "
                f"AND [System.TeamProject] = 'DevFusion2'"
            )
            wiql_query = {"query": wiql}
            wiql_url = base_url + "wit/wiql?api-version=7.2-preview.2"
            response = requests.post(wiql_url, headers=headers, json=wiql_query)
            response.raise_for_status()
            result = response.json()
            work_item_ids = [item["id"] for item in result.get("workItems", [])]

            if not work_item_ids:
                return f"No bug found with title '{query.title}'."
            if len(work_item_ids) > 1:
                return f"Multiple bugs found with title '{query.title}'. Please specify the bug ID."

            # Fetch details for the single matching bug
            bug_url = base_url + f"wit/workitems/{work_item_ids[0]}?fields={','.join(fields)}&api-version=7.2-preview.3"
            response = requests.get(bug_url, headers=headers)
            response.raise_for_status()
            bug_data = response.json()
            fields_data = bug_data["fields"]
        else:
            return "Error: Either bug_id or title must be provided."

        # Format the response
        description = clean_html(fields_data.get("System.ReproSteps", "N/A"))
        output = f"Bug {fields_data.get('System.Id')}:\n"
        output += f"  Title: {fields_data.get('System.Title', 'N/A')}\n"
        output += f"  Repro Steps: {description}\n"
        output += f"  State: {fields_data.get('System.State', 'N/A')}\n"
        output += f"  Assigned To: {fields_data.get('System.AssignedTo', {}).get('displayName', 'Unassigned')}\n"
        output += f"  Priority: {fields_data.get('Microsoft.VSTS.Common.Priority', 'N/A')}\n"
        output += f"  Created Date: {fields_data.get('System.CreatedDate', 'N/A')}\n"
        output += f"  Changed Date: {fields_data.get('System.ChangedDate', 'N/A')}"
        return output

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching bug details: {str(e)}")
        return f"Error fetching bug details: {str(e)}"


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

# Deadline Tools
class DeadlineQuery(BaseModel):
    work_item_id: Optional[str] = Field(default=None, description="Specific work item ID to check deadline for")
    check_all: bool = Field(default=False, description="If true, check all user stories for finish dates")


@tool
def check_azure_finish_dates(query: DeadlineQuery) -> str:
    """Checks Azure DevOps user stories for Microsoft.VSTS.Scheduling.FinishDate."""
    logger.info(f"Invoking check_azure_finish_dates with query: {query}")
    base_url, headers = get_azure_api_config()
    current_date = datetime.now()
    threshold_days = 3

    wiql = "SELECT [System.Id], [Microsoft.VSTS.Scheduling.FinishDate] FROM WorkItems WHERE [System.WorkItemType] = 'User Story' AND [System.TeamProject] = 'DevFusion2'"
    if query.work_item_id:
        wiql += f" AND [System.Id] = '{query.work_item_id}'"
    wiql_query = {"query": wiql}
    wiql_url = base_url + "wit/wiql?api-version=7.2-preview.2"

    try:
        response = requests.post(wiql_url, headers=headers, json=wiql_query)
        response.raise_for_status()
        result = response.json()
        work_item_ids = [item["id"] for item in result.get("workItems", [])]

        if not work_item_ids:
            return f"No user stories found{' for work item ' + query.work_item_id if query.work_item_id else ''}."

        fields = ["System.Id", "System.Title", "Microsoft.VSTS.Scheduling.FinishDate"]
        ids_str = ",".join(map(str, work_item_ids))
        work_items_url = base_url + f"wit/workitems?ids={ids_str}&fields={','.join(fields)}&api-version=7.2-preview.3"
        detail_response = requests.get(work_items_url, headers=headers)
        detail_response.raise_for_status()
        work_items = detail_response.json()["value"]

        results = []
        for item in work_items:
            fields = item.get("fields", {})
            story_id = fields.get("System.Id")
            title = fields.get("System.Title", "N/A")
            finish_date_str = fields.get("Microsoft.VSTS.Scheduling.FinishDate")

            if not finish_date_str:
                results.append(f"User Story {story_id} ('{title}'): No finish date set.")
                continue

            try:
                # Azure returns dates in ISO format (e.g., '2025-04-05T00:00:00Z')
                finish_date = datetime.strptime(finish_date_str, "%Y-%m-%dT%H:%M:%SZ")
                time_diff = (finish_date - current_date).days
                if query.check_all and time_diff > threshold_days:
                    continue  # Skip non-approaching deadlines unless specific ID is queried
                status = (f"past due by {-time_diff} days" if time_diff < 0 else
                          "due today" if time_diff == 0 else
                          f"{time_diff} day{'s' if time_diff > 1 else ''} left")
                results.append(
                    f"User Story {story_id} ('{title}'): Finish date {finish_date.strftime('%Y-%m-%d')} ({status})")
            except ValueError:
                results.append(f"User Story {story_id} ('{title}'): Invalid finish date format.")

        if not results:
            return "No approaching deadlines found within the next 3 days."
        return "\n".join(results)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Azure finish dates: {str(e)}")
        return f"Error fetching Azure finish dates: {str(e)}"


# Weekly Status Tools
class WeeklyQuery(BaseModel):
    period: str = Field(default="current", description="Time period: 'current' for this week, 'next' for next week")
    team: Optional[str] = Field(default=None, description="Specific team name if applicable")


from report_pdf_generator import weekly_report
@tool
def generate_weekly_status(query: WeeklyQuery) -> str:
    """Generates a weekly status report based on the specified period."""
    return weekly_report("")

class NonClosedStoriesQuery(BaseModel):
    select: list[str] = Field(
        default=["System.Id", "System.Title", "System.State", "System.AssignedTo"],
        description="List of fields to select for non-closed user stories"
    )
    limit: Optional[int] = Field(default=None, description="Maximum number of items to return")


@tool
def fetch_non_closed_user_stories(query: NonClosedStoriesQuery) -> str:
    """Fetches all user stories that are not in 'Closed' state and generates a concise text summary."""
    logger.info(f"Invoking fetch_non_closed_user_stories with query: {query}")
    base_url, headers = get_azure_api_config()


    wiql = "SELECT [System.Id] FROM WorkItems WHERE [System.WorkItemType] = 'User Story' AND [System.State] != 'Closed' AND [System.TeamProject] = 'DevFusion2'"
    wiql_query = {"query": wiql}
    wiql_url = base_url + "wit/wiql?api-version=7.2-preview.2"

    try:
        response = requests.post(wiql_url, headers=headers, json=wiql_query)
        response.raise_for_status()
        result = response.json()
        work_item_ids = [item["id"] for item in result.get("workItems", [])]

        if not work_item_ids:
            return "No non-closed user stories found."

        if query.limit:
            work_item_ids = work_item_ids[:query.limit]

        fields = query.select
        ids_str = ",".join(map(str, work_item_ids))
        work_items_url = base_url + f"wit/workitems?ids={ids_str}&fields={','.join(fields)}&api-version=7.2-preview.3"
        detail_response = requests.get(work_items_url, headers=headers)
        detail_response.raise_for_status()
        work_items = detail_response.json()["value"]

        summary = []
        for item in work_items:
            fields = item.get("fields", {})
            story_id = fields.get("System.Id", "N/A")
            title = fields.get("System.Title", "No Title")
            state = fields.get("System.State", "N/A")
            assigned_to = fields.get("System.AssignedTo", {}).get("displayName", "Unassigned")
            summary.append(f"ID {story_id}: '{title}' ({state}, Assigned to: {assigned_to})")

        return "Summary of Non-Closed User Stories:\n" + "\n".join(
            summary) if summary else "No non-closed user stories to summarize."
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching non-closed user stories: {str(e)}")
        return f"Error fetching non-closed user stories: {str(e)}"

@tool
def wait_for_manager_confirmation(tool_input: str) -> str:
    """Waits for manager confirmation via Telegram."""
    global application
    try:
        data = json.loads(tool_input)
        manager_message = data["manager_message"]
        manager_contact = data["manager_contact"]
        response_event = threading.Event()
        user_response = [None]
        sent_message = application.bot.send_message(chat_id=GROUP_CHAT_ID, text=manager_message)
        message_id = sent_message.message_id

        def handle_manager_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if update.message.reply_to_message and update.message.reply_to_message.message_id == message_id:
                user_response[0] = update.message.text.lower()
                response_event.set()

        handler = MessageHandler(filters.TEXT & ~filters.COMMAND & filters.REPLY, handle_manager_response)
        application.updater.dispatcher.add_handler(handler)
        response_event.wait(timeout=300)
        application.updater.dispatcher.remove_handler(handler)
        return user_response[0] or json.dumps({"error": "No response received"})
    except Exception as e:
        logger.error(f"Error in wait_for_manager_confirmation: {str(e)}")
        return json.dumps({"error": str(e)})


@tool
def confirm_update_finish_date(data: dict) -> str:
    """Confirms and updates the finish date for a user story."""
    return "Finish date updated."


# Control Tools
class CompleteOrEscalate(BaseModel):
    cancel: bool = True
    reason: str


class ToAzureAssisstant(BaseModel):
    request: str = Field(description="Fetch all the data and answer the user's question.")


class ToDeadlineAssistant(BaseModel):
    request: str = Field(description="Check deadline datasat and answer the user's question.")


class ToWeeklyAssistant(BaseModel):
    request: str = Field(description="Generate weekly status or planning data and answer the user's question.")


# Agents
class ScrumMasterAgent:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        result = self.runnable.invoke(state)
        if not result.tool_calls and (
                not result.content or
                (isinstance(result.content, str) and result.content.startswith("<tool-use>"))
        ):
            messages = state["messages"] + [HumanMessage(content="Respond with a real output.")]
            state = {**state, "messages": messages}
            result = self.runnable.invoke(state)
        return {"messages": result}


def azureinfo_agent_node(state: State, config: RunnableConfig):
    try:
        logger.info(f"Azure agent processing state: {state}")
        result = azureinfo_runnable.invoke(state)
        logger.info(f"Azure agent LLM output: {result}")
        if not result.tool_calls and (
                not result.content or
                isinstance(result.content, str) and result.content.startswith("<tool-use>")
        ):
            messages = state["messages"] + [HumanMessage(
                content="Please use the fetch_azure_board_data tool to check for missing values or retrieve data as requested.")]
            state = {**state, "messages": messages}
            result = azureinfo_runnable.invoke(state)
            logger.info(f"Azure agent re-invoked with fallback: {result}")
        return {"messages": result}
    except Exception as e:
        logger.error(f"Error in azureinfo_agent_node: {str(e)}")
        return {
            "messages": [
                AIMessage(
                    content=f"Error processing Azure request: {str(e)}. Please clarify your request or try again.")
            ]
        }


def deadline_agent_node(state: State, config: RunnableConfig):
    if not state.get("dialog_state"):
        state["dialog_state"] = ["deadline_agent"]
    result = deadline_runnable.invoke(state)
    if not result.tool_calls and (
            not result.content or
            isinstance(result.content, str) and result.content.startswith("<tool-use>")
    ):
        # Check if user is asking for previous results
        last_message = state["messages"][-1].content.lower()
        if "show me the result" in last_message and state.get("deadline_results"):
            return {"messages": AIMessage(content=state["deadline_results"])}
        messages = state["messages"] + [
            HumanMessage(content="Please use the check_azure_finish_dates tool to fetch approaching deadlines.")]
        state = {**state, "messages": messages}
        result = deadline_runnable.invoke(state)
    # Store tool output in state if it's a deadline check
    if result.tool_calls and result.tool_calls[0]["name"] == "check_azure_finish_dates":
        state["deadline_results"] = None  # Will be updated by ToolMessage
    last_message = state["messages"][-1]
    if isinstance(last_message,
                  ToolMessage) and last_message.content and last_message.name == "check_azure_finish_dates":
        state["deadline_results"] = last_message.content
        return {"messages": AIMessage(content=last_message.content)}
    return {"messages": result}


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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.2,
                             api_key=os.getenv("GEMINI_API_KEY"))

scrum_master_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional corporate multi-language Scrum Master Agent for a big multi national company. Greet everyone politely and show gratitude in every answer. "
               "If the user asks about Azure Boards, mentions user story, or requests 'missing values in new user stories', delegate to the Azure agent. "
               "If the user asks about deadlines or due dates, delegate to the Deadline agent. "
               "If the user asks about weekly status, planning, or sprint progress, delegate to the Weekly agent."),
    ("placeholder", "{messages}")
])
scrum_master_runnable = scrum_master_prompt | llm.bind_tools(
    [ToAzureAssisstant, ToDeadlineAssistant, ToWeeklyAssistant],
    tool_choice="auto"
)

from langchain_core.prompts import ChatPromptTemplate
azureinfo_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AzureInfo agent responsible for performing CRUD operations and summarizing user stories on Azure Boards. "
     "You MUST use the provided tools and MUST NOT generate fabricated responses. Based on the user's request, use the appropriate tool:\n"
     "- For checking missing values in 'New' user stories, call fetch_azure_board_data with check_missing_in_new=True.\n"
     "- For reading data (e.g., 'show all active user stories'), call fetch_azure_board_data with appropriate filters.\n"
     "- For summarizing all non-closed user stories (e.g., 'summarize non-closed user stories'), call fetch_non_closed_user_stories.\n"
     "- For fetching user stories in 'QA Fail' state with their child bugs (e.g., 'show QA Fail stories with bugs'), call fetch_qa_fail_stories_with_bugs.\n"
     "- For fetching details of a specific bug (e.g., 'get details for bug ID 123'), call get_bug_details with the bug ID.\n"
     "- For creating a work item, call create_azure_work_item.\n"
     "- For updating a work item, call update_azure_work_item.\n"
     "- For deleting a work item, call delete_azure_work_item.\n"
     "If the request is unclear, ask for clarification and do not proceed without a tool call. "
     "Return the tool's output directly unless clarification is needed. DO NOT fabricate data. Always rely on the tool's response."),
    ("placeholder", "{messages}")
])
azureinfo_runnable = azureinfo_prompt | llm.bind_tools(
    [
        fetch_azure_board_data,
        fetch_non_closed_user_stories,
        fetch_qa_fail_stories_with_bugs,  # New tool
        get_bug_details,                  # New tool
        create_azure_work_item,
        update_azure_work_item,
        delete_azure_work_item,
        CompleteOrEscalate
    ],
    tool_choice="auto"
)

deadline_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Scrum assistant integrated with Azure DevOps, specializing in managing user story deadlines and facilitating team communication. "
     "Process queries by identifying intent: checking deadlines, extending finish dates, updating status, completing user stories, or handling QA testing. "
     "Use tools promptly and accurately:\n"
     "- For deadline checks, call check_azure_finish_dates.\n"
     "- For status updates, call fetch_azure_data and process_json_data.\n"
     "- For completion, call update_azure_values.\n"
     "- For extensions, call confirm_update_finish_date or update_azure_values based on prior extensions.\n"
     "- For QA, use wait_for_user_response and update_azure_values.\n"
     "Extract details flexibly (e.g., user story ID, emails). If details are missing, request clarification. "
     "Log all actions and errors. Respond concisely and professionally."),
    ("placeholder", "{messages}")
])
deadline_runnable = deadline_prompt | llm.bind_tools(
    [
        check_azure_finish_dates,
        wait_for_manager_confirmation,
        confirm_update_finish_date,
        CompleteOrEscalate
    ],
    tool_choice="auto"
)

weekly_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Weekly agent. When a user asks about weekly status, planning, or sprint progress, "
               "use the generate_weekly_status tool. Construct the WeeklyQuery based on the user's request:\n"
               "- For 'what’s the status for this week', use period='current'.\n"
               "- For 'show next week’s plan', use period='next'.\n"
               "- For 'weekly status for Team A', use period='current', team='Team A'.\n"
               "If the query is unclear, ask for clarification. Present the tool's output directly."),
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
azure_tools = [
    fetch_azure_board_data,
    fetch_non_closed_user_stories,
    fetch_qa_fail_stories_with_bugs,
    get_bug_details,
    create_azure_work_item,
    update_azure_work_item,
    delete_azure_work_item
]
builder.add_node("azure_tools", create_tool_node_with_fallback(azure_tools))
builder.add_node("enter_deadline_agent", create_entry_node("Deadline Assistant", "deadline_agent"))
builder.add_node("deadline_agent", deadline_agent_node)
deadline_tools = [check_azure_finish_dates, wait_for_manager_confirmation,
                  confirm_update_finish_date]
builder.add_node("deadline_tools", create_tool_node_with_fallback(deadline_tools))
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

# Telegram Bot Setup
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROUP_CHAT_ID = "-4702403141"  # Update with your chat ID
application = None


# Function to process user input through the workflow
async def process_message(message: str, user_id: str) -> str:
    logger.info(f"Processing message from user {user_id}: '{message}'")
    state = {
        "messages": [HumanMessage(content=message)],
        "user_info": f"User ID: {user_id}",
        "deadline_results": None
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
            elif isinstance(last_message, AIMessage) and last_message.content and not last_message.content.startswith(
                    "<tool-use>"):
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
    await update.message.reply_text(
        "Hello! I'm your Scrum Master bot. Let me check for approaching deadlines and missing values in user stories with state 'new'.")

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
    global application
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    tz = timezone('Asia/Kolkata')
    scheduler = AsyncIOScheduler(timezone=tz)

    # Helper function to split messages
    def split_message(text, max_length=4096):
        lines = text.split('\n')
        messages = []
        current_message = ""
        for line in lines:
            if len(current_message) + len(line) + 1 > max_length:
                messages.append(current_message)
                current_message = line
            else:
                if current_message:
                    current_message += '\n' + line
                else:
                    current_message = line
        if current_message:
            messages.append(current_message)
        return messages

    # Updated daily_update function
    async def daily_update():
        tz = timezone('Asia/Kolkata')
        current_time = datetime.now(tz).strftime('%I:%M %p %Z')
        logger.info(f"Running daily update at {current_time}")

        missing_values_query = "Check for missing 'description', 'acceptance criteria' and 'assign To' in user stories with state 'new'"
        missing_values_response = await process_message(missing_values_query, "automated_task")

        deadline_query = "Check for approaching deadlines"
        deadline_response = await process_message(deadline_query, "automated_task")

        non_closed_query = "summarize non-closed user stories"
        non_closed_response = await process_message(non_closed_query, "automated_task")

        # Send header message
        await application.bot.send_message(chat_id=GROUP_CHAT_ID, text=f"🌅 **Daily Update at {current_time}**")

        # Send missing values section
        missing_values_text = "❌ **Missing Values in New User Stories**:\n" + missing_values_response
        missing_values_messages = split_message(missing_values_text)
        for msg in missing_values_messages:
            await application.bot.send_message(chat_id=GROUP_CHAT_ID, text=msg)

        # Send approaching deadlines section
        deadline_text = "⏰ **Approaching Deadlines**:\n" + deadline_response
        deadline_messages = split_message(deadline_text)
        for msg in deadline_messages:
            await application.bot.send_message(chat_id=GROUP_CHAT_ID, text=msg)

        # Send user stories summary section
        summary_text = "📋 **User Stories Summary**:\n" + non_closed_response
        summary_messages = split_message(summary_text)
        for msg in summary_messages:
            await application.bot.send_message(chat_id=GROUP_CHAT_ID, text=msg)

        logger.info(f"Daily update sent to chat {GROUP_CHAT_ID}")
    # Schedule daily update at 9:00 AM IST
    scheduler.add_job(daily_update, 'cron', hour=16, minute=32,second=0, timezone=tz)
    scheduler.start()
    logger.info("Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
