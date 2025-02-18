from health_check.health_check import OllamaClientHealth
from olama.ollama_client import OllamaClient

class Initiator:
    def __init__(self):
        self.ollamaClientHealth = OllamaClientHealth()
        self.ollamaClient = OllamaClient()
        pass

    def main(self):
        if(self.ollamaClientHealth.healthCheck() == False):
            print("Ollama is not running")
            return
        else:
            print("Ollama is running")
            self.ollamaClient.main()