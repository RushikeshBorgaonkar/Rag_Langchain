from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

model = ChatGroq(model="llama3-8b-8192")
messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi. Isnt it a beautiful day!"),
]

response = model.invoke(messages)
print(response)
parser = StrOutputParser()

result = model.invoke(messages)

response = parser.invoke(result)

print(response)

chain = model | parser  

result1 = chain.invoke(messages)
result = chain.stream(messages)
print(result1)
print(result)



system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

result = prompt_template.invoke({"language": "japanese", "text": "hi"})

print(result)
result.to_messages()

chain = prompt_template | model | parser

result = chain.invoke({"language": "japanese", "text": "hi ,this is a awesome day"})

print(result)

