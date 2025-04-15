import os
from langchain_community.llms import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv

load_dotenv() 
APIKEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = APIKEY

# temperature is degree of hallucation
llm = OpenAI(temperature=0) 

search = DuckDuckGoSearchResults()
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for when youneed to answer questions about current events. You should ask targeted questions",
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    # ZERO SHOT REACT DESsCRIPTION in langchain is a langchain angent does a reasoning step before aciting
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

try:
    response = agent.run("What is the current price of bitcoin?")
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")