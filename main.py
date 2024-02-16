import llm

def get_api_key(key_name):
    file_path = "api.txt"
    api_keys = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            api_keys[key.strip()] = value.strip()

    return api_keys[key_name]

model = llm.get_model("gpt-3.5-turbo")
model.key = get_api_key("gpt-key")
print(model.key)
response = model.prompt("Five surprising names for a pet pelican")
print(response.text())