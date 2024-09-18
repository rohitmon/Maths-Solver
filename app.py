import streamlit as st    
from langchain_groq import ChatGroq
from langchain.chains import LLMChain , LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities  import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import  Tool , initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant")
st.title("Text to Math Problem Solver using Google Gemma 2")

groq_api_key=st.sidebar.text_input("Groq API Key" , type="password")

if not groq_api_key:
    st.info("Please enter your Groq API Key")
    st.stop()

llm=ChatGroq(model_name="Gemma2-9b-It", groq_api_key=groq_api_key)

wikipedia_wrapper=WikipediaAPIWrapper()

wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Tool for searching the internet to find the various info on the topics mentioned."
)

math_chain=LLMMathChain.from_llm(llm=llm)

calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Tool to perform basic arithmetic operations like addition, subtraction, multiplication, and division. Only mathematical input question needs to be provided."
)

prompt = """
You are an agent tasked for solving user mathematical questions. Logically arrive at the solution and display it pointwise for the question below
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain=LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="Tool for reasoning about the given context and answering questions."
)

assistant_agent=initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        { 'role' : 'assistant' , 'content' : "Hi  , I'm a math chatbot who can answer all your math related questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

## function to generate the response
def generate_response(question):
    response = assistant_agent.invoke({'input' : question})
    return response

question = st.text_area("Enter your question: " , "")
## Let's start the interaction
if st.button("find my answer"):
    if question:
        with st.spinner("Gnererating response..."):
            st.session_state.messages.append({'role' : 'user' , 'content' : question})
            st.chat_message('user').write(question)
            st_cb = StreamlitCallbackHandler(st.container() , expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({'role' : "assistant" , 'content' : response})
            st.write("## Response :")
            st.success(response)
    else:
        st.warning("Please enter a question")