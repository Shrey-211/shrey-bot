import json
import requests
import yaml

class OllamaClientHealth:
    def __init__(self):
        with open('health_check/ollama_config.yml', 'r') as file:
            data = yaml.safe_load(file)
        self.chat_endpoint = data['OLLAMA_BASE']
        self.headers = {"Content-Type": "application/json"}

    def healthCheck(self):
        try:
            with requests.get(self.chat_endpoint, headers=self.headers) as response:
                status = response.raise_for_status()
                if(response.text == None):
                    # print("Error: Health check failed with status", status)
                    # print("Response:", response.text)
                    return False
                else:
                    # print("Health check passed")
                    return True
        except requests.Timeout:
            # print("Error: Request timed out")
            return False

# Testing purpose only
def main():
    clientHealth = OllamaClientHealth()
    clientHealth.healthCheck()

if __name__ == "__main__":
    main()
