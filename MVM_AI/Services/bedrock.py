import boto3
import json

def call_bedrock_endpoint(Question, mensaje):  # ya migrado
   
       
    file_path = 'SystemPrompt.json'

    # Leer el archivo JSON
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    systemPrompJson= data[0]["SystemP"]
    
    system_prompt=[{"text" : f"{systemPrompJson}"}]
    
    bedrock_client = boto3.client(service_name='bedrock-runtime')
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"    
    
    message = [{"role": "user", "content": [{"text": f"System_Prompt: {systemPrompJson}.\n Pregunta: {Question}. \n Curriculum :{mensaje}"}]}]

    
       
    temperature = 0.4
    top_k = 100
    inference_config = {"temperature": temperature}
    # Additional inference parameters to use.
    additional_model_fields = {"top_k": top_k}
    
    response = bedrock_client.converse(
        modelId=model_id,
        messages=message,
        system=system_prompt,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields,
        
        )
    output_message = response['output']['message']['content'][0]['text']   
    
    return output_message

