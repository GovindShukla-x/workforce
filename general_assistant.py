import os, re
import json
import logging
from typing import List, Optional, Tuple
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Initialize embeddings
embedding = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)
class ConversationTurn(BaseModel):
    role: str
    content: str

def load_config():
    with open('config.json', 'r') as config_file:
        return json.load(config_file)

def get_action_and_prompt(persona_name: str, skill_name: str, intent_name: str) -> Tuple[Optional[str], Optional[str]]:
    config = load_config()
    for persona in config['personas']:
        if persona['name'] == persona_name:
            for skill in persona['skills']:
                if skill['name'] == skill_name:
                    for intent in skill['intents']:
                        if intent['name'] == intent_name:
                            return intent['action'], intent['system_prompt']
    return None, None

async def perform_analysis(
    persona: str,
    skill: str,
    prompt: str,
    question: str,
    conversation_history: List[ConversationTurn],
    websocket,
    input_file_unique_id: str,
    output_file_unique_id: str,
    image_base64: Optional[str] = None,
    speech_mode: Optional[bool] = False,
    custom_instruction: Optional[str] = None
):
    logger.info(f"Performing analysis based on: {question}")
    logger.info(f"Input ChromaDB instance (knowledge base): {input_file_unique_id}")
    logger.info(f"Output ChromaDB instance (questions): {output_file_unique_id}")

    try:
        # Initialize ChromaDB instances
        knowledge_base = Chroma(
            persist_directory=f"chroma_db/{input_file_unique_id}",
            embedding_function=embedding,
            collection_name=f"{input_file_unique_id}"
        )

        #chck if the output file exists, if no, then answer the question from the user using the 
        questions_db = Chroma(
            persist_directory=f"chroma_db/{output_file_unique_id}",
            embedding_function=embedding,
            collection_name=f"{output_file_unique_id}"
        )

        # Retrieve all questions from the output ChromaDB instance
        all_docs = questions_db.get()
        logger.info(f"Questions in the output ChromaDB instance: {all_docs}")
        
        # Extract questions from the 'documents' field and split them
        if isinstance(all_docs, dict) and 'documents' in all_docs:
            questions_text = all_docs['documents'][0] if all_docs['documents'] else ""
        else:
            questions_text = ""
        # questions = [doc.page_content for doc in all_docs]
        questions = re.split(r'\n+|\s*\d+\.\s*', questions_text)
        questions = [q.strip() for q in questions if q.strip()]

        if not questions:
            # If no questions found in the output ChromaDB, use the user's question
            questions = [question]

        logger.info(f"Processed questions: {questions}")

        # Get action and system prompt
        # action, system_prompt = get_action_and_prompt(persona, skill, "analyze")

        system_prompt = f"""
        You are an AI assistant specializing in {persona} with a focus on {skill}. 
        Your task is to analyze the given question and context, and provide a comprehensive answer.
        """

        # Process each question
        for idx, question in enumerate(questions):
            logger.info(f"Processing question {idx + 1}: {question}")

            # Perform similarity search in the knowledge base to find relevant information
            search_results = knowledge_base.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in search_results])

            # Generate response using the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]

            response = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=messages,
                temperature=1,
                max_tokens=4096,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            answer = response.choices[0].message.content

            await websocket.send_json({
                "chunk": f"Query:{question}\nAnswer:{answer}",
                "end": False
            })

            # Send the question and answer directly through the websocket
        #     await websocket.send_json({
        #         "question": question,
        #         "answer": answer,
        #         "progress": f"{idx + 1}/{len(questions)}",
        #         "end": False
        #     })

        # # Send completion message
        # await websocket.send_json({
        #     "message": "Analysis completed",
        #     "end": True
        # })
        await websocket.send_json({
                "chunk": f"Task Completed",
                "end": True})

        return "Analysis completed successfully"

    except Exception as e:
        error_message = f"An error occurred during analysis: {str(e)}"
        logger.error(error_message)
        await websocket.send_json({
            "error": error_message,
            "end": True
        })
        return None

async def other_action(
    persona: str,
    skill: str,
    prompt: str,
    question: str,
    conversation_history: List[ConversationTurn],
    websocket,
    input_file_unique_id: Optional[str] = None,
    output_file_unique_id: Optional[str] = None,
    image_base64: Optional[str] = None,
    speech_mode: Optional[bool] = False,
    custom_instruction: Optional[str] = None
):
    logger.info(f"Generating response for: {question}")
    logger.info(f"Using knowledge base: {input_file_unique_id}")

    try:
        # Initialize ChromaDB instance for the knowledge base
        knowledge_base = Chroma(
            persist_directory=f"chroma_db/{input_file_unique_id}",
            embedding_function=embedding,
            collection_name=f"{input_file_unique_id}"
        )

        # Perform similarity search in the knowledge base to find relevant information
        search_results = knowledge_base.similarity_search(question, k=3)
        logger.info(f"Search results: {search_results}")
        context = "\n".join([doc.page_content for doc in search_results])

        logger.info(f"Context retrieved from the knowledge base: {context}")

        system_prompt = f"""
        You are an AI assistant specializing in {persona} with a focus on {skill}. 
        Your task is to analyze the given question and context, and provide a comprehensive answer.
        {custom_instruction if custom_instruction else ''}
        """

        # Generate response using the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=messages,
            temperature=1,
            max_tokens=4096,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                await websocket.send_json({
                    "chunk": content,
                    "end": False
                })

        if not full_response:
            await websocket.send_json({
                "chunk": "I apologize, but I couldn't generate a response. Please try asking your question again.",
                "end": False
            })

        

        await websocket.send_json({
            "chunk": "",
            "end": True
        })

        await websocket.send_json({
            "chunk": full_response,
            "end": True
        })
        

        logger.info(f"Response generated successfully")
        return full_response

    except Exception as e:
        error_message = f"An error occurred while generating the response: {str(e)}"
        logger.error(error_message)
        await websocket.send_json({
            "error": error_message,
            "end": True
        })
        return None
# from crewai import Agent, Task, Crew, Process
# from crewai_tools import BaseTool

# class KnowledgeBaseTool(BaseTool):
#     name: str = "Knowledge Base Query Tool"
#     description: str = "Queries the knowledge base for relevant information"

#     def __init__(self, knowledge_base):
#         self.knowledge_base = knowledge_base

#     def _run(self, query: str) -> str:
#         results = self.knowledge_base.similarity_search(query, k=3)
#         return "\n".join([doc.page_content for doc in results])

# async def perform_analysis_agent(
#     persona: str,
#     skill: str,
#     prompt: str,
#     question: str,
#     conversation_history: List[ConversationTurn],
#     websocket,
#     input_file_unique_id: str,
#     output_file_unique_id: str,
#     image_base64: Optional[str] = None,
#     speech_mode: Optional[bool] = False,
#     custom_instruction: Optional[str] = None
# ):
#     logger.info(f"Performing analysis based on: {question}")
#     logger.info(f"Input ChromaDB instance (knowledge base): {input_file_unique_id}")
#     logger.info(f"Output ChromaDB instance (questions): {output_file_unique_id}")

#     try:
#         # Initialize ChromaDB instances
#         knowledge_base = Chroma(
#             persist_directory=f"chroma_db/{input_file_unique_id}",
#             embedding_function=embedding,
#             collection_name=f"{input_file_unique_id}"
#         )
#         questions_db = Chroma(
#             persist_directory=f"chroma_db/{output_file_unique_id}",
#             embedding_function=embedding,
#             collection_name=f"{output_file_unique_id}"
#         )

#         # Retrieve all questions from the output ChromaDB instance
#         all_docs = questions_db.get()
#         questions = [doc.page_content for doc in all_docs]

#         # Get action and system prompt
#         action, system_prompt = get_action_and_prompt(persona, skill, "analyze")
#         if not system_prompt:
#             system_prompt = f"""
#             You are an AI assistant specializing in {persona} with a focus on {skill}. 
#             Your task is to analyze the given question and context, and provide a comprehensive answer.
#             """

#         # Create the analysis agent
#         analysis_agent = Agent(
#             role=f"{persona} Specialist",
#             goal=f"Provide accurate and comprehensive answers based on the knowledge base",
#             backstory=f"You are an AI assistant with expertise in {persona}, focusing on {skill}. Your task is to analyze questions and provide insightful answers.",
#             verbose=True,
#             allow_delegation=False,
#             tools=[KnowledgeBaseTool(knowledge_base)]
#         )

#         # Create tasks for each question
#         tasks = [
#             Task(
#                 description=f"Analyze and answer the following question: {question}",
#                 agent=analysis_agent
#             )
#             for question in questions
#         ]

#         # Create the crew
#         analysis_crew = Crew(
#             agents=[analysis_agent],
#             tasks=tasks,
#             verbose=2,
#             process=Process.sequential
#         )

#         # Process each task and send results through websocket
#         for idx, task in enumerate(analysis_crew.tasks):
#             result = await analysis_crew.kickoff(inputs={'task': task})
            
#             await websocket.send_json({
#                 "question": questions[idx],
#                 "answer": result,
#                 "progress": f"{idx + 1}/{len(questions)}",
#                 "end": False
#             })

#         # Send completion message
#         await websocket.send_json({
#             "message": "Analysis completed",
#             "end": True
#         })

#         return "Analysis completed successfully"

#     except Exception as e:
#         error_message = f"An error occurred during analysis: {str(e)}"
#         logger.error(error_message)
#         await websocket.send_json({
#             "error": error_message,
#             "end": True
#         })
#         return None


