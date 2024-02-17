import llm
import google.generativeai as genai

def get_api_key(key_name):
    file_path = "api.txt"
    api_keys = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            api_keys[key.strip()] = value.strip()

    return api_keys[key_name]

def prompt_chatGPT(text):
    model = llm.get_model("gpt-3.5-turbo")
    model.key = get_api_key("gpt-key")
    response = model.prompt(text)
    return response.text()


def prompt_googleai(text):
    genai.configure(api_key=get_api_key("google-ai-key"))
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(text)
    return response.text




