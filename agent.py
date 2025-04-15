import os 
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool 
from langchain.agents import AgentType
from langchain.tools import DuckDuckGoSearchRun 
APIKEY = os.getenv("APIKEY")

os.environ["OPENAI_API_KEY"] = APIKEY

# temperature is degree of hallucation
llm = OpenAI(temperature=0) 

search = DuckDuckGoSearchRun()
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