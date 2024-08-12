# %%
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_aws import ChatBedrock

# %%
def demo_chatbot():
    max_gen_len = 128
    temperature = 0.1
    top_p = 0.9
    demo_llm = ChatBedrock(
        credentials_profile_name="mohsen-bedrock",
        model_id="meta.llama3-70b-instruct-v1:0",
        model_kwargs={
            "max_gen_len": max_gen_len,
            "temperature": temperature,
            "top_p": top_p
        },
        region_name="us-east-1"
    )
    return demo_llm

# %%
def demo_memory():
    llm_data = demo_chatbot()
    memory = ConversationSummaryBufferMemory(llm=llm_data,max_token_limit=300)
    return memory

# %%
def demo_conversation(input_text,memory):
    llm_chain_data = demo_chatbot()
    llm_conversation = ConversationChain(
        llm=llm_chain_data,
        memory=memory,
        verbose=True
    )
    chat_reply = llm_conversation.invoke(input_text)
    return chat_reply

# %%
demo_conversation("tell me about your self ",demo_memory())

# %%



