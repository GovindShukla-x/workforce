
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import logging

from typing import Optional, List
import openai
from pydantic import BaseModel
from typing import Optional, List
from langchain_community.vectorstores import  Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai.chat_models.azure import AzureChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from flask import Flask, request, jsonify



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)







# Configure OpenAI API key and version
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize Azure services

# llm = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_KEY"),
#                         azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
#                         api_version = os.getenv("AZURE_OPENAI_API_VERSION")
#                      )


llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=api_version,
    api_key=os.getenv("AZURE_OPENAI_KEY"), 
)
embedding = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_KEY")

        )


def setup_azure_embedding_client():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_KEY")
    )

embedding_function = setup_azure_embedding_client()


class ConversationTurn(BaseModel):
    role: str
    content: str

class IntentResponse(BaseModel):
    intent: str
    standalone_question: str

class StandaloneQuestionResponse(BaseModel):
    standalone_question: str

async def stream_response(response: str, websocket):
    chunk_size = 10  # Adjust this value to control the chunk size
    for i in range(0, len(response), chunk_size):
        chunk = response[i:i+chunk_size]
        await websocket.send_json({
            "chunk": chunk,
            "end": False
        })
    
    await websocket.send_json({
        "chunk": "",
        "end": True
    })
    
    await websocket.send_json({
        "full_message": response,
        "end": True
    })


class ConversationTurn(BaseModel):
    role: str
    content: str

async def create_job_description(persona: str, skill: str, prompt: str, question: str, conversation_history: List[ConversationTurn], websocket, input_file_unique_id: Optional[str] = None, output_file_unique_id: Optional[str] = None, image_base64: Optional[str] = None, speech_mode: Optional[bool] = False, custom_instruction: Optional[str] = None):
    logger.info(f"Creating job description based on: {question}")
    logger.info(f"Persona: {persona}, Skill: {skill}")
    
    system_message = {
        "role": "system",
        "content": f"""You are an AI assistant specializing in {persona} working at bank with expertise in {skill}. 
        Your task is to create a comprehensive job description based on the provided information.
        
        Please include the following sections in the job description:
        1. Job Title
        2. Job Summary
        3. Roles and Responsibilities
        4. Required Qualifications
        5. Preferred Qualifications

        {custom_instruction if custom_instruction else ""}
        Every section to have comprehensive list of bullet points, sorted by importance.
        """
    }
    
    messages = [system_message]
    # if conversation_history:
    #     messages += [{"role": turn.role, "content": turn.content} for turn in conversation_history[-3:]]
    messages.append({"role": "user", "content": f"Please create a detailed job description based on this information: {question}"})
    logger.info(f"Messages: {messages}")
    
    try:
        completion = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=messages,
            temperature=1,
            max_tokens=4096,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )
        
        full_message = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                full_message += delta
                await websocket.send_json({
                    "chunk": delta,
                    "end": False
                })
        
        await websocket.send_json({
            "chunk": "",
            "end": True
        })
        
        await websocket.send_json({
            "full_message": full_message,
            "end": True
        })
        
        if output_file_unique_id:
            with open(f"{output_file_unique_id}_job_description.txt", "w") as f:
                f.write(full_message)
            logger.info(f"Job description saved to {output_file_unique_id}_job_description.txt")
        
        return full_message
    
    except Exception as e:
        error_message = f"An error occurred while creating the job description: {str(e)}"
        logger.error(error_message)
        await websocket.send_json({
            "error": error_message,
            "end": True
        })
        return None

async def prepare_interview_questions(persona: str, skill: str, prompt: str, question: str, conversation_history: List[ConversationTurn], websocket, input_file_unique_id: Optional[str] = None, output_file_unique_id: Optional[str] = None, image_base64: Optional[str] = None, speech_mode: Optional[bool] = False, custom_instruction: Optional[str] = None):
    # Implement interview material preparation logic here
    logger.info(f"Preparing interview materials based on: {question}")
    # Log all the input parameters
    logger.info(f"Persona: {persona}")
    logger.info(f"Skill: {skill}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Question: {question}")
    logger.info(f"Conversation History: {conversation_history}")
    logger.info(f"Input File Unique ID: {input_file_unique_id}")
    logger.info(f"Output File Unique ID: {output_file_unique_id}")
    # logger.info(f"Image Base64: {image_base64}")
    logger.info(f"Speech Mode: {speech_mode}")
    logger.info(f"Custom Instruction: {custom_instruction}")

    response = f"Preparing interview materials for: {question}"
    await stream_response(response, websocket)



async def screen_job_applicant(persona: str, skill: str, prompt: str, question: str, conversation_history: List[ConversationTurn], websocket, input_file_unique_id: Optional[str] = None, output_file_unique_id: Optional[str] = None, image_base64: Optional[str] = None, speech_mode: Optional[bool] = False, custom_instruction: Optional[str] = None):
    # Implement interview material preparation logic here
    # logger.info(f"Evaluating job applicant based on: {question}")
    # # Log all the input parameters
    # logger.info(f"Persona: {persona}")
    # logger.info(f"Skill: {skill}")
    # logger.info(f"Prompt: {prompt}")
    # logger.info(f"Question: {question}")
    # logger.info(f"Conversation History: {conversation_history}")
    # logger.info(f"Input File Unique ID: {input_file_unique_id}")
    # logger.info(f"Output File Unique ID: {output_file_unique_id}")
    # # logger.info(f"Image Base64: {image_base64}")
    # logger.info(f"Speech Mode: {speech_mode}")
    # logger.info(f"Custom Instruction: {custom_instruction}")

    # response = f"Evaluating job applicant for: {question}"

    # unique_id1 = "569bf0e2-71a2-4411-a8d6-0b8f69a1b29e"
    # unique_id2 = "ab29a358-7ee4-4b35-9645-9c1bdeae73f0"


    # # Initialize ChromaDB instances
    # jd_vectorstore = Chroma(persist_directory=f"chroma_db/{unique_id1}", embedding_function=embedding, collection_name=f"{unique_id1}")
    # cv_vectorstore = Chroma(persist_directory=f"chroma_db/{unique_id2}", embedding_function=embedding, collection_name=f"{unique_id2}")

    # cv_retrievar = cv_vectorstore.as_retriever()
    # jd_retreivar = jd_vectorstore.as_retriever()

    # print("intialized vector stoe")
    # results = cv_vectorstore.similarity_search("skills", k=1)

    # print(f"{results[0].page_content}")

    # skills_string = cv_retrievar.get_relevant_documents('key skills')

    # print(f"skills_string is {skills_string}")

    # # Custom Tools
    # class CVRetrievalTool(BaseTool):
    #     name: str = "CV Retrieval Tool"
    #     description: str = "Retrieves relevant CV information based on a query"

    #     def _run(self, query: str, *, config: dict = None) -> str:
    #         # Use the 'config' argument if needed
    #         # results = cv_vectorstore.similarity_search(query, k=4)
    #         # return results[0].page_content if results else "No relevant information found in CV."\
    #         return cv_retrievar.get_relevant_documents(query)

    # class JDRetrievalTool(BaseTool):
    #     name: str = "Job Description Retrieval Tool"
    #     description: str = "Retrieves relevant job description information based on a query"

    #     def _run(self, query: str, *, config: dict = None) -> str:
    #         # Use the 'config' argument if needed
    #         # results = jd_vectorstore.similarity_search(query, k=4)
    #         # return results[0].page_content if results else "No relevant information found in Job Description."
    #         return jd_retreivar.get_relevant_documents(query)

    # # Agents

    # tools = [CVRetrievalTool(), JDRetrievalTool()]
    # cv_analyst = Agent(
    #     role='CV Analyst',
    #     goal='Analyze the candidate\'s CV and extract relevant information',
    #     backstory="You are an experienced HR professional with expertise in analyzing CVs.",
    #     verbose=True,
    #     allow_delegation=False,
    #     tools=[CVRetrievalTool()],
    #     llm=llm
    # )

    # jd_analyst = Agent(
    #     role='Job Description Analyst',
    #     goal='Analyze the job description and identify key requirements',
    #     backstory="You are an experienced recruiter with deep understanding of job requirements.",
    #     verbose=True,
    #     allow_delegation=False,
    #     tools=[JDRetrievalTool()],
    #     llm=llm
    # )

    # skills_matcher = Agent(
    #     role='Skills Matcher',
    #     goal='Compare the candidate\'s skills with job requirements',
    #     backstory="You are an AI specialized in matching candidate skills to job requirements.",
    #     verbose=True,
    #     allow_delegation=False,
    #     llm=llm
    # )

    # experience_evaluator = Agent(
    #     role='Experience Evaluator',
    #     goal='Evaluate the candidate\'s experience against job requirements',
    #     backstory="You are an AI with expertise in assessing professional experience.",
    #     verbose=True,
    #     allow_delegation=False,
    #     llm=llm
    # )

    # report_generator = Agent(
    #     role='Report Generator',
    #     goal='Compile a comprehensive report on the candidate\'s suitability for the job',
    #     backstory="You are an AI specialized in creating clear, concise, and informative reports.",
    #     verbose=True,
    #     allow_delegation=False,
    #     llm=llm
    # )

    # # Tasks
    # task1 = Task(
    #     description="Analyze the candidate's CV and extract key information including skills, experience, and education.",
    #     agent=cv_analyst,
    #     expected_output="A comprehensive summary of the candidate's background.",
    #     tools=[CVRetrievalTool()]
    # )

    # task2 = Task(
    #     description="Analyze the job description and identify key requirements including required skills, experience, and qualifications.",
    #     agent=jd_analyst,
    #     expected_output="A detailed list of job requirements.",
    #     tools=[JDRetrievalTool()]

    # )

    # task3 = Task(
    #     description="Compare the candidate's skills with the job requirements and identify matches and gaps.",
    #     agent=skills_matcher,
    #     expected_output="A comparison of candidate skills vs job requirements.",
    #     context=[task1, task2]
    # )

    # task4 = Task(
    #     description="Evaluate the candidate's experience against the job requirements.",
    #     agent=experience_evaluator,
    #     expected_output="An assessment of how well the candidate's experience matches the job requirements.",
    #     context=[task1, task2]
    # )

    # task5 = Task(
    #     description="Compile a final report summarizing the candidate's suitability for the job, including skills match, experience evaluation, and overall recommendation.",
    #     agent=report_generator,
    #     expected_output="A table on the candidate's suitability for the job considering matching skills, experience and leadership.",
    #     context=[task1, task2, task3, task4]
    # )

    # # Create Crew
    # crew = Crew(
    #     agents=[cv_analyst, jd_analyst, skills_matcher, experience_evaluator, report_generator],
    #     tasks=[task1, task2, task3, task4, task5],
    #     verbose=2
    # )

    # # Execute the crew
    # result = crew.kickoff()

    # print("Final Report:")
    # # print(result)
    result = "Evaluating job applicant for: {question}"
    print(result)

    await stream_response(result, websocket)


