import os
import discord
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition 

# -----------------------------------------
# 1. SETUP LANGGRAPH AGENT
# -----------------------------------------
# Ensure your API key is correct
os.environ["GOOGLE_API_KEY"] = ""

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def simple_calculator(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

tools = [simple_calculator]

# Use a stable model currently supported by the API
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools=[simple_calculator])

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", tool_node) 

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot") 

app = graph_builder.compile()

# -----------------------------------------
# 2. SETUP DISCORD BOT
# -----------------------------------------
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Agent is online and logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    async with message.channel.typing():
        # Pass the message to the agent
        response = await app.ainvoke({"messages": [("user", message.content)]})
        
        # Access the content directly using the .content attribute
        # which is the standard way to retrieve text from recent LangChain models
        answer = response["messages"][-1].content
        
        # If content is a list (multimodal), join the text parts
        if isinstance(answer, list):
            answer = "".join([item.get("text", "") for item in answer if isinstance(item, dict)])
            
        await message.channel.send(answer)

# 3. Run the application
DISCORD_TOKEN = ""
client.run(DISCORD_TOKEN)
