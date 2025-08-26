from huggingface_hub import whoami
from dotenv import load_dotenv

load_dotenv()
print(whoami())
