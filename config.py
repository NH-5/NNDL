import dotenv
import os

dotenv.load_dotenv(override=True)

BARK_KEY = os.getenv('BARK_KEY')