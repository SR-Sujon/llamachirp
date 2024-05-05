# Import necessary modules and classes
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.callbacks.tracers.wandb import WandbTracer
import chainlit as cl
from chainlit.input_widget import Select
import PyPDF2
from chainlit.input_widget import Slider
# Define a function to execute when the conversation starts
@cl.on_chat_start
async def on_chat_start():
    # Initialize variable to store uploaded files
    files = None 
    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Hi I'm Lexi, your AI assistant. Please upload a PDF file to begin the conversation!",
            accept=["application/pdf"],
            max_size_mb=60,  # Optionally limit the file size
            timeout=180,  # Set a timeout for user response
            author="Lexi",
            type="assistant_message",
        ).send()

        settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Llama - Model",
                values=["codegemma","codellama","gemma","llama2","llama3","llama3-gradient","dolphin-llama3", "llava","mistral", "phi3", "stablelm2"],
                initial_index=4,
                )
            ]
        ).send()
        selected_model = settings["Model"]
    # Get the first uploaded file
    file = files[0]

    # Print the file object for debugging
    print(file)

    # Sending Image for displaying status
    elements1 = [
    cl.Image(name="image", display="inline", path="wait.png",size="medium")
    ]
    elements2 = [
    cl.Image(name="image", display="inline", path="ready.png", size="medium")
    ]
    
    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing `{file.name}`...",type="user_message", elements=elements1)
    await msg.send()

    
    
    # Read the PDF file
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(
            model=selected_model,
            temperature=0.7,
            system="You are an expert research assistant who will analyze, correlate, and extract relevant information from the given context and answer questions asked by the user. Your output should be precise and accurate with source information."
        ),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. Your system is ready. Now, please select a MODEL from the options below and ask questions! Note: By default the model is set to Llama2."
    msg.elements = elements2
    await msg.update()

    # Store the chain in the user session
    cl.user_session.set("chain", chain)


# Define a function to execute upon receiving a message
@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")
    
    # Initialize an asynchronous callback handler
    cb = cl.AsyncLangchainCallbackHandler()

    # Call the chain with the user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []  # Initialize list to store text elements

    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    
    # Return results
    await cl.Message(content=answer, elements=text_elements).send()
