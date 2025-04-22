import os
import sys
import datetime
import warnings
import json
import textwrap
import requests
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import asyncio
import threading
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ThreadTimeoutError
from PyPDF2 import PdfReader, PdfWriter
from uuid import uuid4

warnings.filterwarnings("ignore")

# Environment Setup
load_dotenv()
AZURE_API_TOKEN = os.getenv("AZURE_API_TOKEN")
GROQ_API_KEY = os.getenv("groq_api")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Azure DevOps Details
ORGANIZATION = os.getenv("ORGANIZATION")
PROJECT = os.getenv("PROJECT")
HEADERS = {
    "Authorization": f"Bearer {AZURE_API_TOKEN}",
    "Content-Type": "application/json"
}

# Date Range (Last 7 Days)
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# Cache File for Insights
INSIGHTS_CACHE_FILE = "scrum_insights_cache.pkl"

# A4 Page Size (in inches)
PAGE_SIZE = (8.27, 11.69)  # A4: 210mm x 297mm

# Styling Constants
PRIMARY_COLOR = '#2c3e50'
SECONDARY_COLOR = '#34495e'
TEXT_COLOR = '#7f8c8d'
TABLE_HEADER_COLOR = '#34495e'
TABLE_CELL_COLOR = '#ecf0f1'

# Telegram Function
def send_pdf_to_telegram(pdf_path: str):
    """Send the generated PDF report to a Telegram group using requests."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        with open(pdf_path, 'rb') as pdf_file:
            files = {'document': pdf_file}
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'caption': f'Scrum Meeting Report ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})'
            }
            response = requests.post(url, files=files, data=data, timeout=60)
        if response.ok:
            print(f"PDF successfully sent to Telegram group (Chat ID: {TELEGRAM_CHAT_ID})")
            return True
        else:
            print(f"Failed to send PDF to Telegram: {response.text}")
            return False
    except Exception as e:
        print(f"Error sending PDF to Telegram: {e}")
        return False

# PDF Compression Function
def compress_pdf(input_path: str, output_path: str):
    """Compress the PDF file to reduce its size."""
    try:
        reader = PdfReader(input_path)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
            page.compress_content_streams()
        with open(output_path, 'wb') as f_out:
            writer.write(f_out)
        print(f"PDF compressed from {os.path.getsize(input_path)} bytes to {os.path.getsize(output_path)} bytes")
        return True
    except Exception as e:
        print(f"Error compressing PDF: {e}")
        return False

# Tool Definition
@tool("scrum_report", return_direct=True)
def scrum_report(tool_input: str = ""):
    """
    Generate a professional Scrum meeting PDF report with Azure DevOps work items and LLM insights.
    User Stories are sorted by state: New, Active, Resolved, Closed.
    """
    def safe_get(data, keys, default="N/A"):
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, default)
            elif isinstance(data, str):
                return data or default
            else:
                return default
        return data or default

    def fetch_azure_data():
        print("Fetching Azure DevOps user stories and bugs...")
        base_url = f"https://dev.azure.com/{ORGANIZATION}/{PROJECT}/_apis/wit/wiql?api-version=7.2-preview.2"

        def fetch_work_items(work_item_type):
            query = f"""
            SELECT [System.Id], [System.WorkItemType], [System.State]
            FROM workitems
            WHERE [System.WorkItemType] = '{work_item_type}'
              AND [System.CreatedDate] >= '{START_DATE}'
              AND [System.CreatedDate] <= '{END_DATE}'
            """
            response = requests.post(base_url, json={"query": query}, headers=HEADERS)
            if not response.ok:
                print(f"Error fetching {work_item_type}: {response.text}")
                return []
            items = response.json().get("workItems", [])
            return [str(item["id"]) for item in items]

        user_story_ids = fetch_work_items("User Story")
        bug_ids = fetch_work_items("Bug")
        all_ids = user_story_ids + bug_ids

        if not all_ids:
            print("No work items found in the last 7 days.")
            return {"user_stories": [], "bugs": []}

        details_url = f"https://dev.azure.com/{ORGANIZATION}/{PROJECT}/_apis/wit/workitems?ids={','.join(all_ids)}&api-version=7.2-preview.2"
        details_response = requests.get(details_url, headers=HEADERS)
        if not details_response.ok:
            print("Failed to fetch work item details.")
            return {"user_stories": [], "bugs": []}

        details_data = details_response.json().get("value", [])
        user_stories = []
        bugs = []
        for item in details_data:
            fields = item.get("fields", {})
            work_item = {
                "id": str(item.get("id")),
                "title": fields.get("System.Title", "N/A"),
                "state": safe_get(fields, ["System.State"], "N/A"),
                "created_date": safe_get(fields, ["System.CreatedDate"], "N/A"),
                "assigned_to": safe_get(fields, ["System.AssignedTo", "displayName"], "Unassigned"),
                "acceptance_criteria": safe_get(fields, ["Microsoft.VSTS.Common.AcceptanceCriteria"], "N/A"),
                "discussion": safe_get(fields, ["System.History"], "N/A"),
                "priority": safe_get(fields, ["Microsoft.VSTS.Common.Priority"], "N/A")
            }
            if fields.get("System.WorkItemType") == "User Story":
                user_stories.append(work_item)
            else:
                bugs.append(work_item)
        return {"user_stories": user_stories, "bugs": bugs}

    def load_insights_cache():
        """Load cached insights from file."""
        try:
            if os.path.exists(INSIGHTS_CACHE_FILE):
                with open(INSIGHTS_CACHE_FILE, 'rb') as f:
                    return pickle.load(f)
            return {"user_stories": {}, "bugs": {}}
        except Exception as e:
            print(f"Error loading insights cache: {e}")
            return {"user_stories": {}, "bugs": {}}

    def save_insights_cache(insights):
        """Save insights to cache file."""
        try:
            with open(INSIGHTS_CACHE_FILE, 'wb') as f:
                pickle.dump(insights, f)
        except Exception as e:
            print(f"Error saving insights cache: {e}")

    def generate_insights(data):
        print("Generating LLM insights for Scrum report...")
        llm = ChatGroq(
            temperature=0.7,
            model="llama3-70b-8192",
            api_key=GROQ_API_KEY
        )
        insights = load_insights_cache()
        prompt_template = PromptTemplate(
            input_variables=["title", "state", "assigned_to", "acceptance_criteria", "discussion", "created_date", "priority"],
            template="""
            You are a Scrum Master analyzing an Azure DevOps work item for a sprint. Provide concise insights:
            - Title: {title}
            - State: {state}
            - Assigned To: {assigned_to}
            - Created Date: {created_date}
            - Priority: {priority}
            - Acceptance Criteria: {acceptance_criteria}
            - Discussion: {discussion}

            Output a structured response with:
            1. **Progress**: Summarize the work item's status and progress (1 sentence).
            2. **Blockers**: Identify any potential issues or delays (1 sentence).
            3. **Recommendations**: Suggest actionable steps for the Scrum team (1 sentence).
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        max_items = 5  # Limit to 5 items for performance

        def process_item(item, category):
            item_id = item['id']
            if item_id not in insights[category]:
                try:
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(chain.invoke, {
                            "title": item['title'],
                            "state": item['state'],
                            "assigned_to": item['assigned_to'],
                            "acceptance_criteria": item['acceptance_criteria'],
                            "discussion": item['discussion'],
                            "created_date": item['created_date'],
                            "priority": item['priority']
                        })
                        response = future.result(timeout=30)
                        insights[category][item_id] = response.get('text', "N/A")
                except ThreadTimeoutError:
                    print(f"Timeout generating insight for item {item_id}")
                    insights[category][item_id] = "Insight generation timed out."
                except Exception as e:
                    print(f"Error generating insight for item {item_id}: {e}")
                    insights[category][item_id] = "Failed to generate insight."

        for category in ["user_stories", "bugs"]:
            for item in data[category][:max_items]:
                process_item(item, category)

        save_insights_cache(insights)
        return insights

    def create_summary(data, insights):
        """Generate a summary for the Scrum meeting report using LLM."""
        llm = ChatGroq(
            temperature=0.7,
            model="llama3-70b-8192",
            api_key=GROQ_API_KEY
        )
        user_stories_count = len(data['user_stories'])
        bugs_count = len(data['bugs'])
        state_counts = {}
        blockers_count = 0
        for item in data['user_stories'] + data['bugs']:
            state = item['state']
            state_counts[state] = state_counts.get(state, 0) + 1
            insight = insights.get('user_stories', {}).get(item['id'],
                      insights.get('bugs', {}).get(item['id'], ""))
            if "Blockers: None" not in insight and "Blockers:" in insight:
                blockers_count += 1

        insights_summary = []
        for category in ["user_stories", "bugs"]:
            for item_id, insight in insights[category].items():
                insights_summary.append(f"Item {item_id}: {insight[:100]}...")

        prompt_template = PromptTemplate(
            input_variables=["user_stories_count", "bugs_count", "state_counts", "blockers_count", "insights_summary"],
            template="""
            You are a Scrum Master preparing a summary for a weekly Scrum meeting. Based on the following data:
            - User Stories: {user_stories_count}
            - Bugs: {bugs_count}
            - State Distribution: {state_counts}
            - Number of Items with Blockers: {blockers_count}
            - Insights Sample: {insights_summary}

            Provide a professional summary (150-200 words) that includes:
            1. **Overview**: Work items completed, in progress, and total (1-2 sentences).
            2. **Key Blockers**: Challenges identified with number of affected items (1-2 sentences).
            3. **Recommendations**: Actionable steps for the next sprint (1-2 sentences).
            Use a formal tone suitable for a Scrum meeting report.
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        try:
            response = chain.invoke({
                "user_stories_count": user_stories_count,
                "bugs_count": bugs_count,
                "state_counts": json.dumps(state_counts),
                "blockers_count": blockers_count,
                "insights_summary": "\n".join(insights_summary[:3])
            })
            return response.get('text', "Summary generation failed.")
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Failed to generate summary."

    def create_pdf_report(data, insights):
        # State order for sorting
        state_order = {'New': 0, 'Active': 1, 'Dev Done': 2, 'QA Pass': 3, 'QA Fail': 4, 'Resolved': 5, 'Closed': 6}
        sorted_user_stories = sorted(data['user_stories'], key=lambda x: state_order.get(x['state'], 7))

        # Aggregate state counts
        states = {'New': 0, 'Active': 0, 'Dev Done': 0, 'QA Pass': 0, 'QA Fail': 0, 'Resolved': 0, 'Closed': 0}
        for item in data['user_stories'] + data['bugs']:
            state = item['state']
            states[state] = states.get(state, 0) + 1
        backlog_state = states

        # File paths with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"scrum_report_{timestamp}.pdf"
        compressed_pdf_path = f"scrum_report_compressed_{timestamp}.pdf"
        page_number = 1

        with PdfPages(pdf_path) as pdf:
            # Cover Page
            fig = plt.figure(figsize=PAGE_SIZE)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.axis('off')
            ax.text(0.5, 0.9, "Scrum Meeting Report", fontsize=26, ha='center', weight='bold', color=PRIMARY_COLOR)
            ax.text(0.5, 0.7, f"{ORGANIZATION} - {PROJECT}", fontsize=20, ha='center', color=SECONDARY_COLOR)
            ax.text(0.5, 0.5, f"Week of {START_DATE} to {END_DATE}", fontsize=16, ha='center', color=TEXT_COLOR)
            ax.text(0.5, 0.3, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    fontsize=12, ha='center', color=TEXT_COLOR)
            ax.text(0.5, 0.1, "Prepared by Grok 3 (xAI)", fontsize=10, ha='center', color=TEXT_COLOR)
            fig.text(0.95, 0.02, f"Page {page_number}", ha='right', fontsize=8, color=TEXT_COLOR)
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0.2)
            plt.close()
            page_number += 1

            # Summary Page
            summary_text = create_summary(data, insights)
            fig = plt.figure(figsize=PAGE_SIZE)
            ax = fig.add_axes([0.05, 0.1, 0.9, 0.85])
            ax.axis('off')
            ax.text(0, 0.95, "Sprint Summary", fontsize=18, ha='left', weight='bold', color=PRIMARY_COLOR)
            wrapped_text = textwrap.fill(summary_text, width=100)
            ax.text(0, 0.9, wrapped_text, fontsize=11, ha='left', va='top')
            fig.text(0.5, 0.05, "Page 2: Executive summary of sprint progress and recommendations",
                     ha='center', fontsize=8, color=TEXT_COLOR)
            fig.text(0.95, 0.02, f"Page {page_number}", ha='right', fontsize=8, color=TEXT_COLOR)
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0.2)
            plt.close()
            page_number += 1

            # Donut Chart for Work Items by State
            fig = plt.figure(figsize=PAGE_SIZE)
            ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
            cmap = plt.get_cmap('Set2')
            states_list = list(backlog_state.keys())
            counts = list(backlog_state.values())
            donut_colors = [cmap(i % 8) for i in range(len(states_list))]
            ax.pie(counts, labels=states_list, autopct='%1.1f%%', startangle=90,
                   wedgeprops=dict(width=0.4), colors=donut_colors, textprops={'fontsize': 10, 'weight': 'bold'})
            ax.set_title("Work Items by State", fontsize=16, pad=20, color=PRIMARY_COLOR)
            fig.text(0.5, 0.05, "Figure 1: Distribution of user stories and bugs by state",
                     ha='center', fontsize=8, color=TEXT_COLOR)
            fig.text(0.95, 0.02, f"Page {page_number}", ha='right', fontsize=8, color=TEXT_COLOR)
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0.2)
            plt.close()
            page_number += 1

            # Horizontal Bar Chart by Member and State
            all_states = list(backlog_state.keys())
            member_state_counts = {}
            for item in data['user_stories'] + data['bugs']:
                assigned_to = item['assigned_to']
                state = item['state']
                if assigned_to not in member_state_counts:
                    member_state_counts[assigned_to] = {s: 0 for s in all_states}
                member_state_counts[assigned_to][state] += 1

            df = pd.DataFrame.from_dict(member_state_counts, orient='index')
            fig = plt.figure(figsize=PAGE_SIZE)
            ax = fig.add_axes([0.15, 0.2, 0.75, 0.7])
            bar_colors = [cmap(i % 8) for i in range(len(all_states))]
            df.plot(kind='barh', stacked=True, ax=ax, color=bar_colors, legend=False)
            total_items = df.sum().sum()
            for bar_group in ax.patches:
                width = bar_group.get_width()
                if width > 0:
                    percentage = (width / total_items) * 100
                    ax.text(bar_group.get_x() + width / 2, bar_group.get_y() + bar_group.get_height() / 2,
                            f'{percentage:.1f}%', ha='center', va='center', color='white', fontsize=9)
            ax.set_title("Work Items by Team Member and State", fontsize=16, pad=20, color=PRIMARY_COLOR)
            ax.set_xlabel("Number of Work Items", fontsize=11)
            ax.set_ylabel("Team Members", fontsize=11)
            ax.legend(title="State", labels=all_states, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            fig.text(0.5, 0.05, "Figure 2: Team-wise breakdown of work items per state",
                     ha='center', fontsize=8, color=TEXT_COLOR)
            fig.text(0.95, 0.02, f"Page {page_number}", ha='right', fontsize=8, color=TEXT_COLOR)
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0.2)
            plt.close()
            page_number += 1

            # Pie Chart for Work Item Types
            user_stories_count = len(data['user_stories'])
            bugs_count = len(data['bugs'])
            fig = plt.figure(figsize=PAGE_SIZE)
            ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
            labels = ['User Stories', 'Bugs']
            sizes = [user_stories_count, bugs_count]
            colors = ['#66b3ff', '#ff9999']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                   textprops={'fontsize': 12, 'weight': 'bold'})
            ax.set_title("Distribution of Work Items", fontsize=16, pad=20, color=PRIMARY_COLOR)
            fig.text(0.5, 0.05, "Figure 3: Proportion of user stories and bugs",
                     ha='center', fontsize=8, color=TEXT_COLOR)
            fig.text(0.95, 0.02, f"Page {page_number}", ha='right', fontsize=8, color=TEXT_COLOR)
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0.2)
            plt.close()
            page_number += 1

            # Table Rendering Function
            def render_table(title, table_data, headers, caption=None, col_widths=None):
                fig = plt.figure(figsize=PAGE_SIZE)
                ax = fig.add_axes([0.05, 0.1, 0.9, 0.85])
                ax.axis('off')
                ax.text(0, 0.98, title, fontsize=16, ha='left', weight='bold', color=PRIMARY_COLOR)
                table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='left',
                                 colColours=[TABLE_HEADER_COLOR] * len(headers), colWidths=col_widths,
                                 bbox=[0, 0.2, 1, 0.75])
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.0, 1.5)
                for (i, j), cell in table.get_celld().items():
                    if i == 0:
                        cell.set_text_props(weight='bold', color='white')
                        cell.set_facecolor(TABLE_HEADER_COLOR)
                    else:
                        cell.set_facecolor(TABLE_CELL_COLOR)
                    cell.set_edgecolor('#bdc3c7')
                    cell.set_height(0.07)
                if caption:
                    fig.text(0.5, 0.05, caption, ha='center', fontsize=8, color=TEXT_COLOR)
                fig.text(0.95, 0.02, f"Page {page_number}", ha='right', fontsize=8, color=TEXT_COLOR)
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.2)
                plt.close()
                return page_number + 1

            # All Work Items Table
            table_data = []
            headers = ["ID", "Title", "State", "Assigned To", "Priority", "Created Date"]
            col_widths = [0.1, 0.3, 0.15, 0.2, 0.1, 0.15]
            for item in sorted_user_stories + data['bugs']:
                table_data.append([
                    item['id'],
                    textwrap.fill(item['title'], width=30),
                    item['state'],
                    item['assigned_to'],
                    item['priority'],
                    item['created_date'][:10]
                ])
            page_number = render_table("All Work Items", table_data, headers,
                                      caption="Table 1: Consolidated list of user stories and bugs",
                                      col_widths=col_widths)

            # Individual State Tables with Insights
            for state in state_order.keys():
                state_data = []
                for item in sorted_user_stories + data['bugs']:
                    if item['state'] == state:
                        insight = insights.get('user_stories', {}).get(item['id'],
                                                                       insights.get('bugs', {}).get(item['id'], "N/A"))
                        state_data.append([
                            item['id'],
                            textwrap.fill(item['title'], width=30),
                            item['assigned_to'],
                            textwrap.fill(insight, width=40)
                        ])
                if state_data:
                    headers = ["ID", "Title", "Assigned To", "LLM Insights"]
                    col_widths = [0.1, 0.3, 0.2, 0.4]
                    page_number = render_table(f"{state} Work Items", state_data, headers,
                                               caption=f"Table: Work items in '{state}' state with LLM insights",
                                               col_widths=col_widths)

        # Handle PDF compression
        if compress_pdf(pdf_path, compressed_pdf_path):
            final_pdf_path = compressed_pdf_path
        else:
            print("Compression failed, using original PDF.")
            final_pdf_path = pdf_path

        # Verify and send PDF
        abs_pdf_path = os.path.abspath(final_pdf_path)
        print(f"PDF report generated at: {abs_pdf_path}")
        if os.path.exists(abs_pdf_path):
            print(f"File exists. Size: {os.path.getsize(abs_pdf_path)} bytes")
            if send_pdf_to_telegram(abs_pdf_path):
                return "Scrum meeting report with LLM insights generated and sent to Telegram successfully."
            else:
                return "Scrum meeting report generated but failed to send to Telegram."
        else:
            return "Error: PDF file not found."

    # Execute the Workflow
    try:
        data = fetch_azure_data()
        insights = generate_insights(data)
        return create_pdf_report(data, insights)
    except Exception as e:
        print(f"Error in report generation: {e}")
        return f"Failed to generate Scrum report: {str(e)}"

# Agent Setup
def create_agent():
    """Create and return the LangChain agent for report generation."""
    llm = ChatGroq(
        temperature=0.7,
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY
    )
    tools = [scrum_report]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Scrum Master assistant that generates professional Scrum meeting reports. Use the provided tool to create reports when requested."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5
    )
    return agent_executor

# Telegram Bot Section
async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /report command by triggering the agent to generate the report."""
    chat_id = str(update.effective_chat.id)
    if chat_id != TELEGRAM_CHAT_ID:
        await update.message.reply_text(f"Sorry, this command is restricted. Received chat_id: {chat_id}")
        return

    await update.message.reply_text("Generating Scrum meeting report, please wait...")

    try:
        start_time = time.time()
        agent_executor = create_agent()
        result = agent_executor.invoke({"input": "Please generate the Scrum meeting report."})
        elapsed_time = time.time() - start_time
        print(f"Report generation took {elapsed_time:.2f} seconds")
        await update.message.reply_text(result["output"])
    except Exception as e:
        print(f"Error in report_command: {e}")
        await update.message.reply_text(f"Failed to generate the report: {str(e)}")

def run_telegram_bot():
    """Run the Telegram bot to listen for commands."""
    print("Starting Telegram bot...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("report", report_command))
    application.run_polling(allowed_updates=Update.ALL_TYPES)

# Run Bot in Background
def run_bot_in_background():
    """Run the Telegram bot in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    run_telegram_bot()

if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot_in_background, daemon=True)
    bot_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script terminated.")