
import tempfile
import asyncio
import os
import shutil
import uuid
from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import json
from langchain_community.document_loaders import PyPDFLoader
from starlette.websockets import WebSocketDisconnect
import docx2txt



from skills import ask, execute_action, determine_intent, ConversationTurn, IntentResponse, generate_standalone_question

import logging
import shutil
import tempfile
import asyncio
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

import weakref

# Add this near the top of your file, after imports
active_connections = weakref.WeakValueDictionary()
class SessionData:
    def __init__(self):
        self.data = {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

router = APIRouter(
    tags=["Skill"],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error - da skills"},
    },
)

UPLOAD_DIR = "uploads"

app = FastAPI()
logger = setup_logging()

origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    # Add any other origins you want to allow
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def setup_azure_embedding_client():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_KEY")
    )

@app.on_event("startup")
async def startup_event():
    global logger
    logger = setup_logging()
    logger.info("Application startup")

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        
        embedding_function = setup_azure_embedding_client()
        unique_id = str(uuid.uuid4())
        vectorstore = Chroma(
            persist_directory=f"chroma_db/{unique_id}",
            embedding_function=embedding_function,
            collection_name=f"{unique_id}"
        )
        
        vectorstore.add_documents(chunks)
        vectorstore.persist()
        
        return {
            "filename": f"{file.filename}",
            "file_unique_id": f"{unique_id}",
            "status": "success",
            "message": f"Successfully uploaded and processed {len(chunks)} chunks into ChromaDB"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)



# @router.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     # Ensure the upload directory exists
#     if not os.path.exists(UPLOAD_DIR):
#         os.makedirs(UPLOAD_DIR)
    
    
#     file_location = os.path.join(UPLOAD_DIR, file.filename)
#     with open(file_location, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     return {"filename": file.filename}

@router.get("/ping")
async def ping():
    return {"message": "pong"}

@router.websocket("/workforce/ai/live-chat")
async def stream(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session_data = SessionData()
    active_connections[session_id] = session_data
    logger.info(f"WebSocket connection established for session {session_id}")
    #Add latter on caapbility to setup a transient vectordb to hold session parameters
    background_tasks = set()
    try:
        while True:
            try:
                request = await websocket.receive_json()
                logger.info(f"Received request: {request}")
                task = asyncio.create_task(compute_genai_response(request, websocket))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
    except Exception as e:
        logger.exception(e)
    finally:
        del active_connections[session_id]
        for task in background_tasks:
            task.cancel()
        try:
            await websocket.close()
        except RuntimeError:
            pass  # WebSocket already closed
        logger.info(f"WebSocket closed for session {session_id}")



import json

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

# async def compute_genai_response(request, websocket):
#     try:
#         #sample structuee of the input json 
#         # {
# 	    #         "id": "1",
# 	    #         "question": "What is the weather like today?",
# 		#         "custom_instruction":"always respond in a friendly and comprehensive manner",
# 	    #         "conversation_history": [
# 	    #             {"role": "user", "content": "Hello!"},
# 	    #             {"role": "assistant", "content": "Hi there! How can I assist you today?"},
# 	    #             {"role": "user", "content": "what is the weather today?"}
# 	    #         ],
# 		# "digital_persona":"HUMANRESOURCES",
# 		# "skill":"CANDIDATEHIRING",
#         # "intent:"JD_CREATION",
# 		# "input_file_name":"file1.txt",
# 		# "file_unique_id":"fc3e5f85-0421-4105-ab1a-558d3b96316e",
# 		# "output_file_name":"file2.text",
# 		# "image_base64":"base64 encoded file"
#         # }
#         req_id = request["id"]
#         user_input = request['question']
#         conversation_history = request['conversation_history']
#         logger.info(f"/stream request user_input len {len(user_input)}")
#         logger.info(f"user input question is {user_input}")
#         logger.info(f"Added to check for deployment")
#         #get the persona and skill from above 
#         persona = request['digital_persona']
#         skill = request['skill']
#         intent = request['intent']
#         # if intent is None:
#             # then call the openai model with macthing persona and skill from teh config json to plrovide the response in form of pydantic object
#             # specifying intent, function and prompt
#         # else:
#             # get the action and prompt from the config json
#             # call the corresponding function as specified by the action from config json
        
#         #abstract below code into the function, also pass the websocket 
        

#         chunks = ask(conversation_history, user_input)
#         full_message = ""
        
#         for chunk in chunks:
#             full_message += chunk
#             await websocket.send_json({
#                 "id": req_id,
#                 "chunk": chunk,
#                 "end": False
#             })
        
#         await websocket.send_json({
#             "id": req_id,
#             "chunk": "",
#             "end": True
#         })
        
#         # Send the full message at the end
#         await websocket.send_json({
#             "id": req_id,
#             "full_message": full_message,
#             "end": True
#         })
        
#     except asyncio.CancelledError as e:
#         logger.exception(e)

async def compute_genai_response(request, websocket):
    try:
        req_id = request["id"]
        user_input = request['question']
        conversation_history = [ConversationTurn(**turn) for turn in request['conversation_history']]
        custom_instruction = request.get('custom_instruction')
        logger.info(f"Conversation history: {conversation_history}")
        persona = request['digital_persona']
        skill = request['skill']
        intent = request.get('intent')
        speech_mode = request.get('speech_mode')
        input_file_unique_id = request.get('input_file_unique_id')
        output_file_unique_id = request.get('output_file_unique_id')
        image_base64 = request.get('image_base64')
        
        logger.info(f"/stream request user_input len {len(user_input)}")
        logger.info(f"user input question is {user_input}")
        logger.info(f"the intent is {intent}")
        
        # if intent is None or intent == "":
        #Add conversation history to come up with intent and the standalone grounded question
        intent_response = await determine_intent(persona, skill, user_input, conversation_history)
        logger.info(f"Intent response: {intent_response}")
        intent = intent_response.intent
        standalone_question = intent_response.standalone_question
        
        logger.info(f"Intent: {intent}, Standalone question: {standalone_question}")
        # return
        #based on persona, skill and intent, get the action and prompt from the config json
        action, prompt = get_action_and_prompt(persona, skill, intent)
        # action = intent_response.actionx
        # prompt = intent_response.prompt
        # else:
        #     logger.info(f"Intent provided: {intent}")
        #     action, prompt = get_action_and_prompt(persona, skill, intent)
        #     standalone_question = await generate_standalone_question(persona, skill, user_input, conversation_history)
        #     logger.info(f"Action and prompt: {action}, {prompt}")
        #     #get the stabdalone question from the user input
        
        if action is None or prompt is None:
            raise ValueError(f"No action or prompt found for persona: {persona}, skill: {skill}, intent: {intent}")
        
        await websocket.send_json({
                "id": req_id,
                "chunk": "Thinking....",
                "end": False
            })
        # await websocket.send_json({
        #     "id": req_id,
        #     "intent": intent,
        #     "action": action,
        #     "end": False
        # })
        #Add input file, out file, file keys, image, speech mode, custom instruction
        await execute_action(persona, skill, action, prompt, standalone_question, conversation_history, websocket, input_file_unique_id, output_file_unique_id, image_base64, speech_mode, custom_instruction)
        
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        await websocket.send_json({
            "id": req_id,
            "error": str(e),
            "end": True
        })

app.include_router(router)



