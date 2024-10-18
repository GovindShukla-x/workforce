from openai import AzureOpenAI
import os
import logging
from dotenv import load_dotenv
import json
# import logging
from pydantic import BaseModel
from typing import Optional, List
import openai
import json
import os

from typing import Optional, List
from openai import AzureOpenAI

from human_resources import create_job_description, prepare_interview_questions, screen_job_applicant
from general_assistant import perform_analysis, other_action



load_dotenv()

client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_KEY"),
                        azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
                        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
                     )



logger = logging.getLogger(__name__)





load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

class ConversationTurn(BaseModel):
    role: str
    content: str

class IntentResponse(BaseModel):
    intent: str
    standalone_question: str

class StandaloneQuestionResponse(BaseModel):
    standalone_question: str


def load_config():
    with open('persona_skills_config.json', 'r') as config_file:
        return json.load(config_file)

def get_action_and_prompt(persona_name, skill_name, intent_name):
    config = load_config()
    for persona in config['personas']:
        if persona['name'] == persona_name:
            for skill in persona['skills']:
                if skill['name'] == skill_name:
                    for intent in skill['intents']:
                        if intent['name'] == intent_name:
                            return intent['action'], intent['system_prompt']
    return None, None

async def generate_standalone_question(persona: str, skill: str, question: str, conversation_history: List[ConversationTurn]) -> StandaloneQuestionResponse:
    config = load_config()
    logging.info(f"persona: {persona}, skill: {skill}, question: {question}")
    
    system_prompt = f""" 
    You are an AI assistant specializing in {persona} with a focus on {skill}. Your task is to analyze the conversation history and the current question to generate a comprehensive standalone question that encapsulates the context and intent of the user's inquiry.

    The standalone question should:
    1. Be self-contained and understandable without additional context
    2. Incorporate relevant information from the conversation history
    3. Be specific to the user's current needs and the {skill} domain
    4. Be phrased in a way that can be answered comprehensively

    Respond with a JSON object containing only the 'standalone_question' field.
    For example: {{"standalone_question": "Based on the company's growth plans and current team structure, what are the key requirements and responsibilities for a new Digital Marketing Specialist role?"}}
    """
    
    conversation_history_str = "\n".join([f"{turn.role}: {turn.content}" for turn in conversation_history[-3:]])
    
    user_prompt = f"""
    Conversation History:
    {conversation_history_str}

    Current Question: {question}

    Based on the conversation history and the current question, generate a comprehensive standalone question that encapsulates the user's intent and provides necessary context for a detailed response.
    """

    logger.info(f"System prompt: {system_prompt}")
    logger.info(f"User prompt: {user_prompt}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=messages,
        temperature=1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"}
    )
    
    response_data = json.loads(response.choices[0].message.content)
    logger.info(f"Generated standalone question: {response_data}")
    
    standalone_question_response = StandaloneQuestionResponse(**response_data)
    return standalone_question_response

async def determine_intent(persona: str, skill: str, question: str, conversation_history: List[ConversationTurn]) -> IntentResponse:
    config = load_config()
    logging.info(f"persona: {persona}, skill: {skill}, question: {question}")
    
    system_prompt = f""" 
    You are an AI assistant specializing in {persona} with a focus on {skill}. Analyze the following question and conversation history to determine the most appropriate intent from the available options and a standalone question. 

    Your task is to:
    1. Determine the most appropriate intent based on the available options.
    2. Generate a comprehensive standalone question that encapsulates the context and intent of the user's inquiry.

    The standalone question should:
    1. Be self-contained and understandable without additional context
    2. Incorporate relevant information from the conversation history
    3. Be specific to the user's current needs and the {skill} domain
    4. Be phrased in a way that can be answered comprehensively

    Respond with a JSON object containing the following fields:
    - 'intent': The name of the most appropriate intent
    - 'standalone_question': A self-contained question that incorporates relevant information from the conversation history and is specific to the user's current needs

    For example: 
    {{
        "intent": "JD Creation",
        "standalone_question": "Based on the company's expansion plans for the marketing team, what are the key requirements, responsibilities, and qualifications for a new Digital Marketing Specialist role focusing on social media?"
    }}
    """
    
    intents = []
    for p in config['personas']:
        if p['name'] == persona:
            for s in p['skills']:
                if s['name'] == skill:
                    intents = s['intents']
                    break
            break
    
    intent_options = "\n".join([f"- {intent['name']}: {intent['description']}" for intent in intents])

    conversation_history_str = "\n".join([f"{turn.role}: {turn.content}" for turn in conversation_history[-3:]])
    
    user_prompt = f"""
    Conversation History:
    {conversation_history_str}

    Current Question: {question}

    Available intents:
    {intent_options}

    Based on the conversation history and the current question, determine the most appropriate intent and generate a comprehensive standalone question that encapsulates the user's intent and provides necessary context for a detailed response.
    """

    logger.info(f"User prompt: {user_prompt}")
    logger.info(f"System prompt: {system_prompt}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"}
    )
    
    intent_data = json.loads(response.choices[0].message.content)
    logger.info(f"Intent data: {intent_data}")
    
    try:
        intent_response = IntentResponse(**intent_data)
        logger.info(f"Intent response: {intent_response}")
        return intent_response
    except ValueError as e:
        logger.error(f"Error parsing intent response: {e}")
        raise ValueError(f"Invalid response format: {intent_data}")



async def get_weather(persona: str, skill: str, prompt: str, question: str, conversation_history: List[ConversationTurn], websocket, input_file_unique_id: Optional[str] = None, output_file_unique_id: Optional[str] = None, image_base64: Optional[str] = None, speech_mode: Optional[bool] = False, custom_instruction: Optional[str] = None):
    # Implement weather information retrieval logic here
    response = f"Getting weather information for: {question}"
    await stream_response(response, websocket)

async def create_investment_plan(persona: str, skill: str, prompt: str, question: str, conversation_history: List[ConversationTurn], websocket, input_file_unique_id: Optional[str] = None, output_file_unique_id: Optional[str] = None, image_base64: Optional[str] = None, speech_mode: Optional[bool] = False, custom_instruction: Optional[str] = None):
    # Implement investment plan creation logic here
    response = f"Creating investment plan based on: {question}"
    await stream_response(response, websocket)

async def create_budget_plan(persona: str, skill: str, prompt: str, question: str, conversation_history: List[ConversationTurn], websocket, input_file_unique_id: Optional[str] = None, output_file_unique_id: Optional[str] = None, image_base64: Optional[str] = None, speech_mode: Optional[bool] = False, custom_instruction: Optional[str] = None):
    # Implement budget plan creation logic here
    response = f"Creating budget plan based on: {question}"
    await stream_response(response, websocket)

async def create_marketing_campaign(persona: str, skill: str, prompt: str, question: str, conversation_history: List[ConversationTurn], websocket, input_file_unique_id: Optional[str] = None, output_file_unique_id: Optional[str] = None, image_base64: Optional[str] = None, speech_mode: Optional[bool] = False, custom_instruction: Optional[str] = None):
    # Implement marketing campaign creation logic here
    response = f"Creating marketing campaign strategy for: {question}"
    await stream_response(response, websocket)

async def generate_content_ideas(persona: str, skill: str, prompt: str, question: str, conversation_history: List[ConversationTurn], websocket, input_file_unique_id: Optional[str] = None, output_file_unique_id: Optional[str] = None, image_base64: Optional[str] = None, speech_mode: Optional[bool] = False, custom_instruction: Optional[str] = None):
    # Implement content idea generation logic here
    response = f"Generating content ideas for: {question}"
    await stream_response(response, websocket)



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

async def execute_action(persona: str, skill: str, action: str, prompt: str, question: str, conversation_history: List[ConversationTurn], websocket, input_file_unique_id: Optional[str] = None, output_file_unique_id: Optional[str] = None, image_base64: Optional[str] = None, speech_mode: Optional[bool] = False, custom_instruction: Optional[str] = None):
    action_functions = {
        "create_job_description": create_job_description,
        "prepare_interview_questions": prepare_interview_questions,
        "get_weather": get_weather,
        "create_investment_plan": create_investment_plan,
        "create_budget_plan": create_budget_plan,
        "create_marketing_campaign": create_marketing_campaign,
        "generate_content_ideas": generate_content_ideas,
        "screen_job_applicant": screen_job_applicant,
        "perform_analysis": perform_analysis,
        "other_action": other_action
    }
    
    if action in action_functions:
        await action_functions[action](persona, skill, prompt, question, conversation_history, websocket, input_file_unique_id, output_file_unique_id, image_base64, speech_mode, custom_instruction)
    else:
        await other_action(persona, skill, prompt, question, conversation_history, websocket, input_file_unique_id, output_file_unique_id, image_base64, speech_mode, custom_instruction)



def ask(conversation_history: list, question: str):
    system_message = {
        "role": "system",
        "content": "Your name is lara croft. you job is to response to  "

    }
    
    logging.info("Deployment Test: This is a new unique log message to verify code changes.")
    
    messages = []
    if conversation_history is not None:
        if len(conversation_history) >= 4:
            logger.info(f"Question len='{len(question)}', trimmed #messages {len(messages)}")
            messages = [system_message] + conversation_history[-3:]
        else:
            messages = [system_message] + conversation_history[:]
    
    logging.info(f"the message is {messages}")
    
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
    
    for chunk in completion:
        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta.content
            if delta is not None and len(delta) > 0:
                yield delta