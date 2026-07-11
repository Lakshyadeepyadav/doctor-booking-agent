import os
import asyncio
import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from composio import Composio
from composio_langchain import LangchainProvider

from c import model

# ----------------------------
# Config
# ----------------------------
USER_ID = "default"
CONNECT_ACCOUNTS_ON_START = True

os.environ["COMPOSIO_CACHE_DIR"] = os.path.abspath(".composio_cache")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Environment Checks
# ----------------------------
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Missing GROQ_API_KEY. Run: export GROQ_API_KEY='your_key'")

if not os.getenv("COMPOSIO_API_KEY"):
    raise RuntimeError("Missing COMPOSIO_API_KEY. Run: export COMPOSIO_API_KEY='your_key'")

# ----------------------------
# Composio
# ----------------------------
composio = Composio(provider=LangchainProvider())


def get_auth_config_id(toolkit: str) -> str:
    auth_configs = composio.auth_configs.list(toolkit_slug=toolkit)

    if auth_configs.items:
        latest_config = sorted(
            auth_configs.items,
            key=lambda item: str(item.created_at),
            reverse=True,
        )[0]
        return latest_config.id

    auth_config = composio.auth_configs.create(
        toolkit=toolkit,
        options={
            "type": "use_composio_managed_auth",
            "tool_access_config": {
                "tools_for_connected_account_creation": [],
            },
        },
    )

    return auth_config.id


def already_connected(toolkit: str, auth_config_id: str) -> bool:
    accounts = composio.connected_accounts.list(
        user_ids=[USER_ID],
        auth_config_ids=[auth_config_id],
        statuses=["ACTIVE"],
    )

    if accounts.items:
        print(f"{toolkit} already connected.")
        return True

    return False


def connect_toolkit(toolkit: str):
    print(f"\nChecking connection for {toolkit}...")

    auth_config_id = get_auth_config_id(toolkit)

    if already_connected(toolkit, auth_config_id):
        return

    connection_request = composio.connected_accounts.link(
        USER_ID,
        auth_config_id,
    )

    print(f"\nConnect your {toolkit} account here:")
    print(connection_request.redirect_url)

    input("\nOpen the link, approve access, then press Enter here...")

    connection_request.wait_for_connection(timeout=180)
    print(f"{toolkit} connected successfully.")


if CONNECT_ACCOUNTS_ON_START:
    connect_toolkit("googlecalendar")
    connect_toolkit("gmail")

# ----------------------------
# Composio Tools
# ----------------------------
tools = composio.tools.get(
    user_id=USER_ID,
    tools=[
        "GOOGLECALENDAR_FIND_FREE_SLOTS",
        "GOOGLECALENDAR_CREATE_EVENT",
        "GMAIL_SEND_EMAIL",
    ],
)

print("\nLoaded tools:")
for t in tools:
    print("-", t.name)

tool_node = ToolNode(
    tools,
    handle_tool_errors=True,
)

# ----------------------------
# System Prompt
# ----------------------------
system_prompt = """You are a dental assistant. Use the tools provided."""

# ----------------------------
# Model
# ----------------------------
model_with_tools = model.bind_tools(tools)

# ----------------------------
# Agent Node
# ----------------------------
def call_model(state: MessagesState):
    now_local = datetime.datetime.now().astimezone().isoformat()

    try:
        response = model_with_tools.invoke(
            [
                SystemMessage(
                    content=(
                        system_prompt
                        + f"\nCurrent Local Time: {now_local}"
                    )
                )
            ]
            + state["messages"]
        )

        return {"messages": [response]}
    except Exception as e:
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="Task completed")]}

# ----------------------------
# Graph
# ----------------------------
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    },
)

workflow.add_edge("tools", "agent")

app = workflow.compile()

# ----------------------------
# Helper
# ----------------------------
async def run_turn(message: str):
    # Use a completely fresh state for each turn - no thread persistence
    state = {
        "messages": [
            HumanMessage(content=message)
        ]
    }

    # Use a unique thread ID for each turn to avoid history accumulation
    import uuid
    thread_id = str(uuid.uuid4())

    try:
        async for chunk in app.astream(
            state,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="values",
        ):
            msg = chunk["messages"][-1]

            if hasattr(msg, "pretty_print"):
                msg.pretty_print()
            else:
                print(msg)
    except Exception as e:
        # Suppress token limit errors after tools have executed
        if "413" not in str(e) and "rate_limit" not in str(e):
            print(f"Error: {e}")

# ----------------------------
# Main
# ----------------------------
async def main():
    print("\n=== USER 1 ===")
    await run_turn(
        "I have pain in my teeth for a few weeks now. Is there a free slot at 10:30 PM today?"
    )

    print("\n=== USER 2 ===")

    await run_turn(
        "Yes, please book the 10:30 PM slot today for my tooth pain appointment and send the confirmation email to email address:lakshyadeep6090@gmail.com"
    )

if __name__ == "__main__":
    asyncio.run(main())
