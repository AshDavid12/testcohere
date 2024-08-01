from langchain.agents import AgentExecutor
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv('.env')

cohere_api_key = os.getenv('COHERE_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')
# Internet search tool - you can use any tool, and there are lots of community tools in LangChain.
# To use the Tavily tool you will need to set an API key in the TAVILY_API_KEY environment variable.
internet_search = TavilySearchResults(api_key= tavily_api_key)

# Create and run the Cohere agent
# Set a Cohere API key in the COHERE_API_KEY environment variable.
llm = ChatCohere()
agent = create_cohere_react_agent(
    llm=llm,
    tools=[internet_search],
    prompt=ChatPromptTemplate.from_template("{question}"),
)
agent_executor = AgentExecutor(agent=agent, tools=[internet_search], verbose=True)

response = agent_executor.invoke({
    "question": "I want to write an essay. Any tips?",
})
# See Cohere's response
print(response.get("output"))
# Cohere provides exact citations for the sources it used
print(response.get("citations"))