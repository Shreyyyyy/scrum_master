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

warnings.filterwarnings("ignore")

# Environment Setup
load_dotenv()
AZURE_API_TOKEN = os.getenv("AZURE_API_TOKEN")
GROQ_API_KEY = os.getenv("groq_api")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not AZURE_API_TOKEN or not GROQ_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("One or more required environment variables are missing.")
    sys.exit(1)

# Azure DevOps Details
ORGANIZATION = "SahejMarwah"
PROJECT = "DevFusion2"
HEADERS = {
    "Authorization": f"Bearer {AZURE_API_TOKEN}",
    "Content-Type": "application/json"
}

# Date Range (Last 7 Days)
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# Cache File for Insights
INSIGHTS_CACHE_FILE = "insights_cache.pkl"

# Telegram Function
def send_pdf_to_telegram(pdf_path: str):
    """Send the generated PDF report to a Telegram group using requests."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        with open(pdf_path, 'rb') as pdf_file:
            files = {'document': pdf_file}
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'caption': f'Weekly Azure DevOps Report ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})'
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
            page.compress_content_streams()  # Compress content streams

        with open(output_path, 'wb') as f_out:
            writer.write(f_out)

        print(f"PDF compressed from {os.path.getsize(input_path)} bytes to {os.path.getsize(output_path)} bytes")
        return True
    except Exception as e:
        print(f"Error compressing PDF: {e}")
        return False
# Tool Definition
@tool("weekly_report", return_direct=True)
def weekly_report(tool_input: str = ""):
    """
    Generate a PDF report with Azure DevOps work items and LLM insights.
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
            print("No work items found in the last 14 days.")
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
                "title": textwrap.shorten(fields.get("System.Title", "N/A"), width=50, placeholder="..."),
                "state": safe_get(fields, ["System.State"], "N/A"),
                "created_date": safe_get(fields, ["System.CreatedDate"], "N/A"),
                "assigned_to": safe_get(fields, ["System.AssignedTo", "displayName"], "Not assigned"),
                "acceptance_criteria": safe_get(fields, ["Microsoft.VSTS.Common.AcceptanceCriteria"], "N/A"),
                "discussion": safe_get(fields, ["System.History"], "N/A")
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

    from langchain_google_genai import ChatGoogleGenerativeAI

    def generate_insights(data):
        print("Generating LLM insights...")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.2,
                                     api_key=os.getenv("GEMINI_API_KEY"))

        insights = load_insights_cache()
        prompt_template = PromptTemplate(
            input_variables=["title", "state", "assigned_to", "acceptance_criteria", "discussion", "created_date"],
            template="""
            Provide insights for the following Azure DevOps work item:
            - Title: {title}
            - State: {state}
            - Assigned To: {assigned_to}
            - Created Date: {created_date}
            - Acceptance Criteria: {acceptance_criteria}
            - Discussion: {discussion}

            Briefly analyze the progress, potential blockers, and provide recommendations.
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        max_items = 3  # Reduced from 10 to speed up

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
                            "created_date": item['created_date']
                        })
                        response = future.result(timeout=30)  # 30-second timeout per item
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

    def create_pdf_report(data, insights):
        state_order = {'New': 0, 'Active': 1,'Dev Done':2,'QA Pass':3,'QA Fail':4,'Resolved': 5, 'Closed': 6}
        sorted_user_stories = sorted(data['user_stories'], key=lambda x: state_order.get(x['state'], 4))

        states = {'New': 0, 'Active': 0,'Dev Done':0,'QA Pass':0,'QA Fail':0,'Resolved':0, 'Closed': 0}
        for item in data['user_stories'] + data['bugs']:
            state = item['state']
            if state in states:
                states[state] += 1
            else:
                states[state] = 1
        backlog_state = states

        # Use timestamp in PDF filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"azure_report_{timestamp}.pdf"
        compressed_pdf_path = f"azure_report_compressed_{timestamp}.pdf"

        with PdfPages(pdf_path) as pdf:
            # Title Page
            fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait: 8.27x11.69 inches
            ax.axis('off')
            ax.text(0.5, 0.7, "Scrum Master Weekly Report", ha='center', va='center', fontsize=24, fontweight='bold')
            ax.text(0.5, 0.6, f"From {START_DATE} to {END_DATE}", ha='center', va='center', fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            # Donut Chart for Work Items by State
            fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait: 8.27x11.69 inches
            cmap = plt.get_cmap('tab10')
            states_list = list(backlog_state.keys())
            counts = list(backlog_state.values())
            donut_colors = [cmap(i % 10) for i in range(len(states_list))]
            ax.pie(counts, labels=states_list, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3),
                   colors=donut_colors, textprops={'fontsize': 12})
            ax.set_title("Work Items by State (Donut Chart)", fontsize=16, pad=20)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            # Horizontal Bar Chart for Work Items by Member and State
            all_states = list(backlog_state.keys())
            member_state_counts = {}
            for item in data['user_stories'] + data['bugs']:
                assigned_to = item['assigned_to'] if item['assigned_to'] else "Not assigned"
                state = item['state']
                if assigned_to not in member_state_counts:
                    member_state_counts[assigned_to] = {s: 0 for s in all_states}
                member_state_counts[assigned_to][state] += 1

            df = pd.DataFrame.from_dict(member_state_counts, orient='index')
            fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait: 8.27x11.69 inches
            bar_colors = [cmap(i % 10) for i in range(len(all_states))]
            bars = df.plot(kind='barh', stacked=True, ax=ax, color=bar_colors)

            total_items = df.sum().sum()
            for i, bar_group in enumerate(bars.containers):
                for bar in bar_group:
                    width = bar.get_width()
                    if width > 0:
                        percentage = (width / total_items) * 100
                        ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2,
                                f'{percentage:.1f}%', ha='center', va='center', color='white', fontsize=9,
                                weight='bold')

            ax.set_title("Work Items by Member and State", fontsize=14, pad=15)
            ax.set_xlabel("Number of Work Items", fontsize=12)
            ax.set_ylabel("Team Members", fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            ax.legend(title="State", labels=all_states, fontsize=10, title_fontsize=11, loc='upper right')
            max_count = df.sum(axis=1).max()
            ax.set_xticks(range(0, int(max_count) + 1, 2))
            plt.tight_layout(pad=2.0)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Table of All Work Items
            fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait: 8.27x11.69 inches
            ax.axis('off')
            ax.set_title("Work Items Table", fontsize=16)

            table_data = []
            headers = ["ID", "Title", "State", "Assigned To", "Created Date"]
            for item in sorted_user_stories:
                table_data.append(
                    [item['id'], item['title'], item['state'], item['assigned_to'], item['created_date'][:10]])
            for item in data['bugs']:
                table_data.append(
                    [item['id'], item['title'], item['state'], item['assigned_to'], item['created_date'][:10]])

            table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='left',
                             colColours=['#f0f0f0'] * len(headers), bbox=[0.05, 0.05, 0.9, 0.9])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.2)
            table.auto_set_column_width([0, 2, 3, 4])
            table.auto_set_column_width([1])
            for (i, j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#d3d3d3')
                cell.set_edgecolor('black')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Individual State Tables
            all_items = sorted_user_stories + data['bugs']
            state_groups = {'New': [], 'Active': [], 'Resolved': [], 'Closed': []}
            for item in all_items:
                state = item['state']
                if state in state_groups:
                    state_groups[state].append(item)

            for state, items in state_groups.items():
                if items:  # Only create a table if there are items for the state
                    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait: 8.27x11.69 inches
                    ax.axis('off')
                    ax.set_title(f"{state} Work Items", fontsize=16)

                    table_data = []
                    headers = ["ID", "Title", "State", "Assigned To", "Created Date"]
                    for item in items:
                        table_data.append(
                            [item['id'], item['title'], item['state'], item['assigned_to'], item['created_date'][:10]])

                    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='left',
                                     colColours=['#f0f0f0'] * len(headers), bbox=[0.05, 0.05, 0.9, 0.9])
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1.2, 1.2)
                    table.auto_set_column_width([0, 2, 3, 4])
                    table.auto_set_column_width([1])
                    for (i, j), cell in table.get_celld().items():
                        if i == 0:
                            cell.set_text_props(weight='bold')
                            cell.set_facecolor('#d3d3d3')
                            cell.set_edgecolor('black')
                        cell.set_edgecolor('black')
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

        # Compress the PDF
        if compress_pdf(pdf_path, compressed_pdf_path):
            final_pdf_path = compressed_pdf_path
        else:
            print("Compression failed, using original PDF.")
            final_pdf_path = pdf_path

        # Send to Telegram (no browser opening)
        abs_pdf_path = os.path.abspath(final_pdf_path)
        print(f"PDF report generated at: {abs_pdf_path}")
        if os.path.exists(abs_pdf_path):
            print(f"File exists. Size: {os.path.getsize(abs_pdf_path)} bytes")
            if send_pdf_to_telegram(abs_pdf_path):
                return "Weekly report with LLM insights generated and sent to Telegram successfully."
            else:
                return "Weekly report generated but failed to send to Telegram."
        else:
            return "Error: PDF file not found."

    # Execute the Workflow
    try:
        data = fetch_azure_data()
        insights = generate_insights(data)
        return create_pdf_report(data, insights)
    except Exception as e:
        print(f"Error in report generation: {e}")
        return f"Failed to generate weekly report: {str(e)}"

# Agent Setup
def create_agent():
    """Create and return the LangChain agent for report generation."""
    llm = ChatGroq(
        temperature=0.7,
        model_name="llama3-70b-8192",
        groq_api_key=GROQ_API_KEY
    )
    tools = [weekly_report]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant that generates Azure DevOps reports. Use the provided tool to create reports when requested. Respond with the tool's output or an error message if the request cannot be processed."),
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
    print(f"Received chat_id: {chat_id}, Expected TELEGRAM_CHAT_ID: {TELEGRAM_CHAT_ID}")
    if chat_id != TELEGRAM_CHAT_ID:
        await update.message.reply_text(
            f"Sorry, this command is restricted to a specific group. Received chat_id: {chat_id}")
        return

    await update.message.reply_text("Generating Azure DevOps report, please wait... This may take a moment.")

    try:
        # Use the agent to process the report request
        start_time = time.time()
        agent_executor = create_agent()
        result = agent_executor.invoke({"input": "Please generate the weekly Azure DevOps report."})
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
    # Start the Telegram bot in a background thread
    bot_thread = threading.Thread(target=run_bot_in_background, daemon=True)
    bot_thread.start()

    # Keep the main thread alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Script terminated.")
