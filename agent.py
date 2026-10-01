import json
import operator
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()


@tool
def estimate_trip_budget(
    days: int,
    nights: int,
    travelers: int,
    hotel_per_night: float,
    food_per_day: float,
    activities_per_day: float,
    transportation_total: float,
) -> str:
    """Estimate a trip budget in USD, using costs for the entire group.

    Supply trip days, hotel nights, and number of travelers separately.
    Hotel is per night; food and all activities combined are per day;
    transportation is for the whole trip. All inputs are required.
    Use positive days and travelers, and nonnegative nights and costs.
    Ask for missing inputs or unclear group costs before calling; do not invent
    prices, assume nights, or treat missing costs as zero. Returns JSON with
    estimated totals and an equal per-person split, covering supplied costs only.
    """
    print(
        f"Trip budget tool called: {days} days, {nights} nights, "
        f"{travelers} travelers"
    )

    breakdown = {
        "hotel": round(nights * hotel_per_night, 2),
        "food": round(days * food_per_day, 2),
        "activities": round(days * activities_per_day, 2),
        "transportation": round(transportation_total, 2),
    }
    total = round(sum(breakdown.values()), 2)
    return json.dumps({
        "currency": "USD",
        "breakdown": breakdown,
        "total": total,
        "per_person": round(total / travelers, 2),
        "per_day": round(total / days, 2),
    })


search_tool = DuckDuckGoSearchRun()
tools = [search_tool, estimate_trip_budget]
tools_by_name = {tool.name: tool for tool in tools}

model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def llm_call(state: MessagesState) -> dict:
    """Call the LLM with the current messages and available tools."""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant that can search the web to answer questions. "
                        "Use the search tool when you need current information."
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: MessagesState) -> dict:
    """Execute tool calls from the last message."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Determine whether to continue to tool execution or end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


def build_agent():
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)

    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph_builder.add_edge("tool_node", "llm_call")

    agent = graph_builder.compile()
    return agent
