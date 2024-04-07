from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

from langchain_core.messages import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

st.set_page_config(page_title="Claude 3 PhD Thesis QA", page_icon="📖")
st.title("📖 Claude 3 PhD Thesis QA")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
#if len(msgs.messages) == 0:
#    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# Set up the LangChain, passing in Message History

model = ChatAnthropic(model='claude-3-opus-20240229')


with open('./PhDFull.txt', 'r') as file:
    thesis = file.read()

system = ( f"Here is an academic paper, a doctoral thesis. <paper>{thesis}</paper>\n\n The user might wants to improve the readability and make adjustments to structure and language to accomodate different audiences or ask questions about the content.." )
human = "{question}"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", human),
    ]
)

chain = prompt | model

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "unused"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response.content)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)