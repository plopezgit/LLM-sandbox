from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import OpenAI
import os
from keys import open_ai_key
from keys import serp_api_key

os.environ['OPEN_API_KEY']= open_ai_key
os.environ["SERPAPI_API_KEY"]= serp_api_key

llm = OpenAI(openai_api_key=open_ai_key)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
agent.invoke("who is the presiden trump wife, and waht is goin to be her age in ten years?")

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.invoke("who is the presiden trump wife, and waht is goin to be her age in ten years?")


#Note:
#LangChainDeprecationWarning: LangChain agents will continue to be supported,
#but it is recommended for new use cases to be built with LangGraph.
#LangGraph offers a more flexible and full-featured framework for building agents,
#including support for tool-calling, persistence of state, and human-in-the-loop workflows.
#For details, refer to the `LangGraph documentation <https://langchain-ai.github.io/langgraph/>`_
#as well as guides for `Migrating from AgentExecutor <https://python.langchain.com/docs/how_to/migrate_agent/>`_
#and LangGraph's `Pre-built ReAct agent <https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/>`_.
