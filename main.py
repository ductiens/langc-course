from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

def main():
  # Test openai
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response}")

    print("Setup complete!")

if __name__ == "__main__":
  main()