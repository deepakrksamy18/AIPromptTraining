import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load the .env file from the workspace's.
load_dotenv(override=True)
required = ["FARM_BASE_URL", "FARM_API_KEY", "FARM_API_VERSION", "MODEL_NAME"]
missing = [name for name in required if not os.getenv(name)]
if missing:
    raise RuntimeError(f"Missing {', '.join(missing)} in .env before running this lab.")
 
model = ChatOpenAI(
    model=os.environ["MODEL_NAME"],
    base_url=f"{os.environ['FARM_BASE_URL']}/openai/deployments/{os.environ['FARM_DEPLOYMENT_NAME']}",
    api_key=os.environ["FARM_API_KEY"],
    default_query={"api-version": os.environ["FARM_API_VERSION"]},
)

extract = ChatPromptTemplate.from_template(
    "Extract only stated symptoms from this report. Do not infer causes. Report: {report}"
) | model | StrOutputParser()

classify = ChatPromptTemplate.from_template(
    "Choose one category: Paint Damage, Fluid Leakage, Braking System, "
    "Electrical System, Human Review. If evidence is insufficient, choose Human Review. "
    "Symptoms: {symptoms}. Return category and supporting evidence."
) | model | StrOutputParser()

report = "Oil is leaking beneath the engine."
symptoms = extract.invoke({"report": report})
result = classify.invoke({"symptoms": symptoms})
print("Extracted symptoms:", symptoms)
print("Classification:", result)