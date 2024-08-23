from Models import model
from fastapi.responses import JSONResponse
from Services.bedrock import call_bedrock_endpoint
from fastapi import APIRouter
import json
import re


router = APIRouter()


def extract_values(text):
    
    response_pattern = r'Response:\s*(.*?)\s*Score:'
    score_pattern = r'Score:\s*(\d+)'
    
    
    response_match = re.search(response_pattern, text, re.DOTALL)
    response = response_match.group(1).strip() if response_match else None
    
    
    score_match = re.search(score_pattern, text)
    score = int(score_match.group(1)) if score_match else None
    
    return response, score


@router.post("/prompt", tags=["ChatRag"])
async def prompt_text(request: model.PromptRequest):
    try:
        Text_CV = request.promptText

        file_path = 'promptread.json'

        # Leer el archivo JSON
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        
        final_json = {"CandidateInfo": {}}
        for entry in data:
            pregunta = entry.get('pregunta', 'No disponible')
            prompt = entry.get('prompt', 'No disponible')
            callOpenAIResponse = call_bedrock_endpoint(prompt,Text_CV)            
            response, score = extract_values(callOpenAIResponse)           

            
            final_json["CandidateInfo"][pregunta] = {
                "response": response,
                "score": f"{score}%"
            }
        return final_json

    except Exception as e:
        return JSONResponse(
            content={
                'success': False,
                'content': str(e),
                'message': 'Error en el prompt'
            },
            status_code=400
        )

