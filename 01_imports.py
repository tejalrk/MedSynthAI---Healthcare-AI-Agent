# ========== CELL 0 ==========
# GPU Setup for Colab
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

# ========== CELL 2 ==========
from getpass import getpass

# safer way (you'll enter token securely)
token = getpass('Enter your GitHub token: ')

# Clone the private repo using https + token
!git clone https://tejalrk:{token}@github.com/UoM-Data-Science-Platforms/uom-msc-cs1.git


# ========== CELL 3 ==========
# !rm -rf /content/uom-msc-cs1/uom-msc-cs1

# ========== CELL 7 ==========
from shutil import copyfile
copyfile('/content/drive/MyDrive/Colab Notebooks/AI_AGENT_kaggle_dataset_final_colab (1).ipynb',
         '/content/uom-msc-cs1/AI_AGENT_kaggle_dataset_final_colab (1).ipynb')

# ========== CELL 11 ==========
# rm -r data
!git add -u output


# ========== CELL 14 ==========
# !git add Healthcare_AI_agent_colab

# ========== CELL 15 ==========
# !git commit -m "Remove Healthcare_AI_agent_colab from repo"

# ========== CELL 16 ==========
# !git push https://tejalrk:{token}@github.com/UoM-Data-Science-Platforms/uom-msc-cs1.git

# ========== CELL 19 ==========
import os
import pandas as pd
import numpy as np
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ========== CELL 20 ==========
 #Pinecone imports
import pinecone
from pinecone import Pinecone, ServerlessSpec

# ========== CELL 21 ==========
# UI imports
import gradio as gr

# Privacy and security
from cryptography.fernet import Fernet
from faker import Faker
import secrets
import string

# Environment setup for Google Colab
from google.colab import userdata

# ========== CELL 22 ==========
# Set your API keys (use Colab secrets for security)
os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
os.environ['VOYAGE_API_KEY'] = userdata.get('VOYAGE_API_KEY')
os.environ['PINECONE_API_KEY'] = userdata.get('PINECONE_API_KEY')

# Verify environment variables
assert os.getenv('ANTHROPIC_API_KEY'), "Anthropic API key not found"
assert os.getenv('VOYAGE_API_KEY'), "Voyage AI API key not found"
assert os.getenv('PINECONE_API_KEY'), "Pinecone API key not found"

print("✅ Environment variables configured successfully")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== CELL 23 ==========
