#importing libraries
import requests
import io
import logging
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END 
from langchain_core.runnables.graph import MermaidDrawMethod 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer 
from IPython.display import display, Image
from databases import ResearchDatabase
from typing import TypedDict
import pprint
import os

# #logging the code
# logger = logging.getLogger(__name__)
# logger.basicConfig(level = logging.INFO)

#loading the dotenv file
load_dotenv()
groqquiz_api_key = os.getenv("RESEARCH_QUESTIONS_GENERATOR")
groqstructure_api_key = os.getenv("RESEARCH_STRUCTURE_GENERATOR")
 
if not groqquiz_api_key:
    raise ValueError("Error loading API Key!")
if not groqstructure_api_key:
    raise ValueError("Error loading API Key!")


#initializing the llm
llm_query = ChatGroq(
    temperature = 0.7,
    api_key = groqquiz_api_key,
    model = "qwen-qwq-32b"
)
llm_structure = ChatGroq(
    temperature = 0.7,
    api_key = groqstructure_api_key,
    model = "qwen-qwq-32b"
)

#testing the model
#print(llm.invoke("Research aboout different cat breeds."))

#defining the state class
class State(TypedDict):
    query: str 
    response: str 

def create_multiple_queries(query):
    prompt = ChatPromptTemplate.from_template("""
        <Goal>
    <Primary>
        To generate five distinct, research-quality search queries that directly address the user's topic. These queries should be formulated as someone would type them into a search engine when conducting thorough academic or professional research.
    </Primary>
</Goal>

<Instructions>
    <Instruction>
        1. Extract the exact concepts and keywords from the user's topic without introducing unrelated themes or expanding the scope unless explicitly requested.
    </Instruction>
    <Instruction>
        2. Preserve the user's specific terminology and focus. Do not substitute or generalize terms (e.g., if user specifies "happiness," do not change to "well-being" or "mental health").
    </Instruction>
    <Instruction>
        3. Create exactly five unique search queries that:
            - Center on the main keywords from the user's input
            - Include thoughtful variations that maintain the original focus
            - Represent different research angles within the stated scope
            - Are formatted as complete search terms someone would actually type
    </Instruction>
    <Instruction>
        4. Structure the output in this JSON format:
        ```json
        {{
            "topic": "USER_INPUT_TOPIC",
            "searchQueries": [
                "QUERY_1",
                "QUERY_2",
                "QUERY_3",
                "QUERY_4",
                "QUERY_5"
            ]
        }}
        ```
    </Instruction>
    <Instruction>
        5. Prioritize depth over breadth in query formulation, focusing on specific aspects that would yield substantive research material.
    </Instruction>
</Instructions>

<Examples>
    <Example>
        <UserInput>gut microbiome link to happiness</UserInput>
        <AgentOutput>
            {{
                "topic": "gut microbiome link to happiness",
                "searchQueries": [
                    "gut microbiome link to happiness research studies",
                    "gut microbial composition and happiness correlation",
                    "gut bacteria influence on happiness levels evidence",
                    "bacterial diversity and reported happiness direct link",
                    "latest findings gut microbiome happiness association"
                ]
            }}
        </AgentOutput>
    </Example>

    <Example>
        <UserInput>Creative writing with AI</UserInput>
        <AgentOutput>
            {{
                "topic": "Creative writing with AI",
                "searchQueries": [
                    "AI creative writing tools",
                    "artificial intelligence creative writing innovations",
                    "AI-assisted fiction writing techniques",
                    "exploring generative AI for creative writing",
                    "case studies AI-driven creative storytelling"
                ]
            }}
        </AgentOutput>
    </Example>

    <Example>
        <UserInput>quantum computing hardware</UserInput>
        <AgentOutput>
            {{
                "topic": "quantum computing hardware",
                "searchQueries": [
                    "quantum computing hardware architecture",
                    "latest developments in quantum processors",
                    "superconducting qubits hardware advancements",
                    "ion trap quantum computing devices",
                    "scalability challenges quantum hardware research"
                ]
            }}
        </AgentOutput>
    </Example>

    <Example>
        <UserInput>big data in retail</UserInput>
        <AgentOutput>
            {{
                "topic": "big data in retail",
                "searchQueries": [
                    "big data retail analytics best practices",
                    "customer insights from big data in retail",
                    "data-driven inventory management retail",
                    "big data solutions for e-commerce personalization",
                    "trends in big data adoption retail sector"
                ]
            }}
        </AgentOutput>
    </Example>

    <Example>
        <UserInput>self-driving car sensors</UserInput>
        <AgentOutput>
            {{
                "topic": "self-driving car sensors",
                "searchQueries": [
                    "lidar vs radar technology in self-driving cars",
                    "self-driving car sensors real-time data processing",
                    "sensor fusion methods for autonomous vehicles",
                    "advancements in self-driving car camera systems",
                    "emerging self-driving sensor technologies market analysis"
                ]
            }}
        </AgentOutput>
    </Example>
</Examples>
The user's topic is: {query}
    """)

    chain = prompt | llm_query
    response = chain.invoke({"query": query})
    return response.content


# query = "research about cats"
# questions = create_multiple_queries(query)
# pprint.pprint(questions)

def  create_researh_structure(response: str):
    """
    Used to create a research structure after the questions have been generated by the agent

    ARGS:
        Response: This is the response of the previous model

    Output:
        The research structure

    """
     
    prompt = ChatPromptTemplate.from_template(
        """
        <Goal>
    <Primary>
        To generate five distinct, research-quality search queries that directly address the user's topic. These queries should be formulated as someone would type them into a search engine when conducting thorough academic or professional research.
    </Primary>
</Goal>

<Instructions>
    <Instruction>
        1. Extract the exact concepts and keywords from the user's topic without introducing unrelated themes or expanding the scope unless explicitly requested.
    </Instruction>
    <Instruction>
        2. Preserve the user's specific terminology and focus. Do not substitute or generalize terms (e.g., if user specifies "happiness," do not change to "well-being" or "mental health").
    </Instruction>
    <Instruction>
        3. Create exactly five unique search queries that:
            - Center on the main keywords from the user's input
            - Include thoughtful variations that maintain the original focus
            - Represent different research angles within the stated scope
            - Are formatted as complete search terms someone would actually type
    </Instruction>
    <Instruction>
        4. Structure the output in this JSON format:
        ```json
        {{
            "topic": "USER_INPUT_TOPIC",
            "searchQueries": [
                "QUERY_1",
                "QUERY_2",
                "QUERY_3",
                "QUERY_4",
                "QUERY_5"
            ]
        }}
        ```
    </Instruction>
    <Instruction>
        5. Prioritize depth over breadth in query formulation, focusing on specific aspects that would yield substantive research material.
    </Instruction>
</Instructions>

<Examples>
    <Example>
        <UserInput>gut microbiome link to happiness</UserInput>
        <AgentOutput>
            {{
                "topic": "gut microbiome link to happiness",
                "searchQueries": [
                    "gut microbiome link to happiness research studies",
                    "gut microbial composition and happiness correlation",
                    "gut bacteria influence on happiness levels evidence",
                    "bacterial diversity and reported happiness direct link",
                    "latest findings gut microbiome happiness association"
                ]
            }}
        </AgentOutput>
    </Example>

    <Example>
        <UserInput>Creative writing with AI</UserInput>
        <AgentOutput>
            {{
                "topic": "Creative writing with AI",
                "searchQueries": [
                    "AI creative writing tools",
                    "artificial intelligence creative writing innovations",
                    "AI-assisted fiction writing techniques",
                    "exploring generative AI for creative writing",
                    "case studies AI-driven creative storytelling"
                ]
            }}
        </AgentOutput>
    </Example>

    <Example>
        <UserInput>quantum computing hardware</UserInput>
        <AgentOutput>
            {{
                "topic": "quantum computing hardware",
                "searchQueries": [
                    "quantum computing hardware architecture",
                    "latest developments in quantum processors",
                    "superconducting qubits hardware advancements",
                    "ion trap quantum computing devices",
                    "scalability challenges quantum hardware research"
                ]
            }}
        </AgentOutput>
    </Example>

    <Example>
        <UserInput>big data in retail</UserInput>
        <AgentOutput>
            {{
                "topic": "big data in retail",
                "searchQueries": [
                    "big data retail analytics best practices",
                    "customer insights from big data in retail",
                    "data-driven inventory management retail",
                    "big data solutions for e-commerce personalization",
                    "trends in big data adoption retail sector"
                ]
            }}
        </AgentOutput>
    </Example>

    <Example>
        <UserInput>self-driving car sensors</UserInput>
        <AgentOutput>
            {{
                "topic": "self-driving car sensors",
                "searchQueries": [
                    "lidar vs radar technology in self-driving cars",
                    "self-driving car sensors real-time data processing",
                    "sensor fusion methods for autonomous vehicles",
                    "advancements in self-driving car camera systems",
                    "emerging self-driving sensor technologies market analysis"
                ]
            }}
        </AgentOutput>
    </Example>
</Examples>
The user's topic is: {query}
    """
    )

    chain = prompt | llm_structure
    response = chain.invoke({'query': query})
    return response.content

# query = "cats"
# structure = create_researh_structure(query)
# print(structure)
query = "Emotional Intelligence In Humanity"
search_response = create_multiple_queries(query)
structure_response= create_researh_structure(query)

#initializing the db
db = ResearchDatabase()
db.insert_research_entry(query, search_response, structure_response)
records = db.fetch_all_entries()
for record in records:
    print(f" ID: {record[0]}, Query: {record[1]}, questions: {record[2]}, Research Structure: {record[3]}, Time: {record[4]}")