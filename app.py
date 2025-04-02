import streamlit as st
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Set up the Streamlit app
st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant", page_icon="π")
st.title("Text To Math Problem Solver using Google Gamma 2")

# Input for Groq API Key
groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API Key to continue")
    st.stop()

# Initialize the LLM (assuming `ChatGroq` exists)
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Initialize the Wikipedia search tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching and interpreting various information on the topic mentioned."
)

# Initialize the Math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math-related questions. Only input mathematical expressions need to be provided."
)

# Prompt template for reasoning and math problem-solving
prompt = """
You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a detailed explanation, displaying it point-wise for the question below.
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all the tools into a chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Reasoning tool for logical questions
reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# Initialize the agent with tools
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # or another available agent type
    verbose=False,
    handle_parsing_errors=True
)

# Session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your math questions."}
    ]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Function to generate the response
def generate_response(user_question):
    response = assistant_agent.invoke({'input': user_question})
    return response

# User input for the math problem
question = st.text_area("Enter your question:"," ")

# Handling button click to find the answer
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({'role': 'assistant', "content": response})
            
            st.write('## Response:')
            st.success(response)
    else:
        st.warning("Please enter the question.")
