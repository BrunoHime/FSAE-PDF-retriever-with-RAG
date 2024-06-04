# LLM
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.globals import set_verbose
set_verbose(True)


# Libraries for the embeddings and the Qdrant database
import qdrant_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai


def retrieve_context(user_input: str,
                     google_api_key: str,
                     top_k: int = 70
                     ) -> str:
    
    """This function retrieves the context of the user's query by connecting with Qdrant database and using cosine similarity
       to find the most similar subtopics.

    Args:
        user_input (str): A question asked by the user about the FSAE rules.
        google_api_key (str): Google API key.
        top_k (int, optional): The number of most similar subtopics to retrieve. Defaults to 25.

    Returns:
        str: Top k most similar subtopics to the user's query.
    """

    # Setting the Google API key
    GOOGLE_API_KEY = google_api_key
    genai.configure(api_key=GOOGLE_API_KEY)

    # Connect with Google's generative AI embeddings using LangChain
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)


    # Embedding of the user's query
    user_vector = embeddings.embed_documents([user_input])[0]

    # Connect with the Qdrant database
    client = qdrant_client.QdrantClient(path="qdrant_path")

    # Top k most similar questions
    similar_subtopics = client.search(
        collection_name="fsae_rules",
        query_vector=user_vector,
        limit=top_k
    )

    context = ""
    for subtopic in similar_subtopics:
        context += f"Subtopic:\n {subtopic.payload['text']}Page extracted:\n {subtopic.payload['page']}\n\n"
    
    return context



def llm_model(user_input: str,
              google_api_key: str,
              top_k: int = 70,
              temperature: float = 0.8) -> dict:

    """This function retrieves the context of the user's query by connecting with Qdrant database and using cosine similarity
       to find the most similar subtopics.

    Args:
        user_input (str): A question asked by the user about the FSAE rules.
        google_api_key (str): Google API key.
        top_k (int, optional): The number of most similar subtopics to retrieve. Defaults to 25.
        temperature (float, optional): The temperature for the LLM model. Defaults to 1.

    Returns:
        str: Returns the answer of the LLM model.
    """

    # Setting the Google API key
    GOOGLE_API_KEY = google_api_key
    genai.configure(api_key=GOOGLE_API_KEY)

    # Setting the generation configuration for the LLM model
    generation_config = {
        'candidate_count': 1,
        'stop_sequences': None,
        'max_output_tokens': None,
        'temperature': temperature,
        'top_p': None,
        'top_k': None,
    }


    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-001",
        generation_config = generation_config,
        google_api_key=google_api_key)
    

    # Template with prompt engineering
    prompt_template = '''
        FSAE competition is a challenging event where teams design and build a formula car to compete in various dynamic and static events.
        Teams must adhere to several rules and regulations to ensure safety and fair competition.

        As an expert in FSAE competition rules, you will answer user questions based on the context provided below.
        If unsure, ask the user for more information, but never provide false information.

        The context below refers to the most similar subtopics from the rules, based on the user's question. Some answers may be found in more than one subtopic,
        so make sure to provide a complete and accurate answer connecting the information from the different subtopics.
        Use only the relevant information from the context.

        
        <CONTEXT>
        {context}

        <OUTPUT FORMAT>
        Respond to the user's question with a clear and detail answer, retrieving the information from the context above. 
        Also provide the source of the information, and a recommendation of a subtopic and its page to find more information.
        Write only the source in italic and start it with, Source:

        Ensure to respond in the same language used by the user.

        <USER'S INPUT> {input}
        '''
    

    # Creating the chain
    prompt = PromptTemplate(
        input_variables=['context', 'input'],
        template=prompt_template
    )

    chain = (
        {'input': RunnablePassthrough(), 'context': RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    context = retrieve_context(user_input=user_input,
                               google_api_key=google_api_key,
                               top_k=top_k)

    answer = chain.invoke({'input': user_input, 'context': context})

    return {"user_input": user_input, "context": context, "answer": answer}
