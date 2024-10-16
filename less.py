import os
import re
import getpass
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from types import SimpleNamespace
from langchain_community.retrievers import WikipediaRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.documents import Document
import google.generativeai as genai
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import requests
import pickle

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, 'dev.json')
PICKLE_PATH = os.path.join(BASE_DIR, 'faiss_data.pkl')



# Load Google API key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# Data Preprocessing
def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    
    sentences = sent_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_sentence = ' '.join([word for word in words if word not in stop_words])
        processed_sentences.append(filtered_sentence)
    
    return ' '.join(processed_sentences)

# Custom data loader
def load_custom_documents(file_path: str) -> List[dict]:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        documents = data.get("data", [])
        print(f"Loaded {len(documents)} documents")
        return documents
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure it's in the correct location.")
        return []
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.")
        return []
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return []


# Vector store setup
def setup_vector_store(documents: List[dict]):
    if os.path.exists(PICKLE_PATH):
        print("Loading pickled FAISS data...")
        with open(PICKLE_PATH, 'rb') as f:
            index, model, processed_docs = pickle.load(f)
        print("Pickled data loaded successfully.")
        return index, model, documents

    print("Processing documents and creating FAISS index...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    processed_docs = []
    vectors = []
    for i, doc in enumerate(documents):
        print(f"Processing document {i}:")
        if 'question' not in doc or 'answer' not in doc:
            print(f"  Missing 'question' or 'answer' field in document at index {i}. Skipping.")
            continue
        
        if not doc['question'] or not doc['answer']:
            print(f"  Empty 'question' or 'answer' field in document at index {i}. Skipping.")
            continue
        
        try:
            processed_question = preprocess_text(doc['question'])
            processed_answer = preprocess_text(doc['answer'])
            combined_text = f"{processed_question} {processed_answer}"
            processed_docs.append(combined_text)
            vector = model.encode(combined_text)
            vectors.append(vector)
            print(f"  Successfully processed document {i}")
        except Exception as e:
            print(f"  Error processing document at index {i}: {str(e)}")
    
    if not processed_docs:
        raise ValueError("No valid documents were processed. Check your input data.")

    dimension = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors))

    print(f"Total processed documents: {len(processed_docs)}")
    
    # Pickle the FAISS index, model, and processed documents
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump((index, model, processed_docs), f)
    print("FAISS data pickled and saved.")

    return index, model, documents

# Model setup
def setup_model():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Pydantic models
class MedicalQuery(BaseModel):
    text: str
    lab_report: Optional[str] = None

class Agent(BaseModel):
    role: str
    goal: str
    backstory: str
    tools: Optional[List[Union[Tool, BaseLanguageModel]]] = Field(default_factory=list)
    llm: Optional[BaseLanguageModel] = None
    verbose: bool = True

    class Config:
        arbitrary_types_allowed = True

class Task(BaseModel):
    description: str
    agent: Agent

    class Config:
        arbitrary_types_allowed = True

# Crew class
class Crew:
    def __init__(self, agents: List[Agent], tasks: List[Task], verbose: int = 2):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose

    def kickoff(self, query: MedicalQuery) -> SimpleNamespace:
        results = []
        for task in self.tasks:
            if self.verbose > 0:
                print(f"Executing task: {task.description}")
            try:
                if task.agent.tools:
                    for tool in task.agent.tools:
                        result = tool.run(self.prepare_query(query))
                        results.append(SimpleNamespace(result=result))
                elif task.agent.llm:
                    result = task.agent.llm.predict(self.prepare_query(query))
                    results.append(SimpleNamespace(result=result))
                else:
                    results.append(SimpleNamespace(result="No tool or LLM found for task"))
            except Exception as e:
                results.append(SimpleNamespace(result=f"Error executing task: {str(e)}"))
        
        corpus = create_corpus(results)
        final_answer = generate_final_answer(self.prepare_query(query), corpus)
        
        return SimpleNamespace(individual_results=results, corpus=corpus, final_answer=final_answer)

    def prepare_query(self, query: MedicalQuery) -> str:
        components = [f"Text query: {query.text}"]
        if query.lab_report:
            components.append(f"Lab report: {query.lab_report}")
        return " | ".join(components)

# Tool functions For WIKIPEDIA
def wikipedia_search(query: str) -> str:
    retriever = WikipediaRetriever()
    docs = retriever.get_relevant_documents(query)
    combined_text = " ".join([doc.page_content for doc in docs])
    words = combined_text.split()
    return " ".join(words[:200])

###Creating Corpus###
def create_corpus(results: List[SimpleNamespace]) -> str:
    return "\n\n".join([result.result for result in results])

###Faiss Agent###
def faiss_qa_search(query: str, index, model, documents, k=5) -> str:
    processed_query = preprocess_text(query)
    query_vector = model.encode([processed_query])
    faiss.normalize_L2(query_vector)
    
    distances, indices = index.search(query_vector, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        doc = documents[idx]
        results.append(f"Question: {doc['question']}\nAnswer: {doc['answer']}\nSimilarity: {distances[0][i]}")
    
    return "\n\n".join(results)

####pmc_site_reterival####

def pmc_open_access_search(query: str, max_results: int = 5) -> str:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi"
    fetch_url = f"{base_url}efetch.fcgi"
    
    search_params = {
        "db": "pmc",
        "term": query,
        "retmax": max_results,
        "format": "json",
        "sort": "relevance"
    }
    
    try:
        search_response = requests.get(search_url, params=search_params)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        id_list = search_data['esearchresult']['idlist']
        
        if not id_list:
            return "No results found."
        
        fetch_params = {
            "db": "pmc",
            "id": ",".join(id_list),
            "rettype": "xml",
            "retmode": "text"
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params)
        fetch_response.raise_for_status()
        
        xml_data = fetch_response.text
        articles = xml_data.split("<article ")
        
        results = []
        for article in articles[1:]:
            title_match = re.search(r'<article-title>(.*?)</article-title>', article)
            abstract_match = re.search(r'<abstract>(.*?)</abstract>', article, re.DOTALL)
            
            title = title_match.group(1) if title_match else "No title available"
            abstract = abstract_match.group(1) if abstract_match else "No abstract available"
            
            abstract = re.sub(r'<[^>]+>', '', abstract)
            abstract = re.sub(r'\s+', ' ', abstract).strip()
            
            results.append(f"Title: {title}\nAbstract: {abstract}\n")
        
        return "\n".join(results)
    except requests.exceptions.RequestException as e:
        return f"Error fetching data from PMC: {str(e)}"

# Setup function for agentic system
def setup_medical_qa_system():
    
    documents = load_custom_documents(JSON_PATH)
    index, model, docs = setup_vector_store(documents)
    llm = setup_model()

    wikipedia_tool = Tool(
        name="Wikipedia Search",
        func=wikipedia_search,
        description="Search for information on Wikipedia"
    )

    faiss_qa_tool = Tool(
        name="FAISS QA Database",
        func=lambda q: faiss_qa_search(q, index, model, docs),
        description="Search for answers in the FAISS-indexed medical dataset."
    )

    pmc_tool = Tool(
        name="PMC Open Access Search",
        func=pmc_open_access_search,
        description="Search for relevant biomedical and life sciences research articles in the PMC Open Access Subset"
    )

    wikipedia_agent = Agent(
        role='Medical Information Researcher',
        goal='Find medical information from Wikipedia',
        backstory='Expert in public medical information with extensive experience in Wikipedia research.',
        tools=[wikipedia_tool],
        verbose=True
    )

    medical_data_agent = Agent(
        role='Medical Data Specialist',
        goal='Find medical data from the local dataset using FAISS',
        backstory='Expert in analyzing and retrieving information from specialized medical databases using vector similarity search.',
        tools=[faiss_qa_tool],
        verbose=True
    )

    pmc_agent = Agent(
        role='PMC Open Access Specialist',
        goal='Find relevant biomedical and life sciences research articles from the PMC Open Access Subset',
        backstory='Expert in searching and analyzing scientific literature from PubMed Central Open Access Subset.',
        tools=[pmc_tool],
        verbose=True
    )

    diagnosis_prompt = """
    You are an AI assistant specializing in medical diagnosis. Your role is to analyze the given information and provide a preliminary diagnostic assessment. Focus on the following departments: Cardiology, Neurology, and Gastroenterology. When presented with a query:

    1. Identify key symptoms, risk factors, and relevant medical history.
    2. Consider potential diagnoses within the specified departments.
    3. Suggest possible diagnostic tests or examinations that could help confirm or rule out these diagnoses.
    4. Highlight any red flags or emergency signs that require immediate medical attention.
    5. Emphasize the importance of professional medical evaluation for a definitive diagnosis.

    Remember: Your goal is to provide informative insights to support the diagnostic process, not to give a final diagnosis. Always maintain a professional tone and prioritize patient safety.

    Based on this guidance, please respond to the following query: {query}
    """

    diagnosis_agent = Agent(
        role='Diagnostic Specialist',
        goal='Provide preliminary diagnostic insights for cardiology, neurology, and gastroenterology',
        backstory='AI with extensive knowledge of diagnostic processes in specific medical departments',
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro"),
        verbose=True
    )

    task_wikipedia = Task(
        description="Find relevant medical information from Wikipedia based on the given question.",
        agent=wikipedia_agent
    )

    task_medical_data = Task(
        description="Find the relevant answer from the medical dataset using FAISS.",
        agent=medical_data_agent
    )

    pmc_task = Task(
        description="Search for relevant biomedical and life sciences research articles in the PMC Open Access Subset.",
        agent=pmc_agent
    )

    task_diagnosis = Task(
        description="Provide preliminary diagnostic insights based on the given medical information.",
        agent=diagnosis_agent
    )

    crew = Crew(
        agents=[wikipedia_agent, medical_data_agent, pmc_agent, diagnosis_agent],
        tasks=[task_wikipedia, task_medical_data, pmc_task, task_diagnosis],
        verbose=2
    )

    return crew
#Generating The Final Assesment
def generate_final_answer(query: str, corpus: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    prompt = f"""
    Based on the following corpus of medical information and the given query, please provide a comprehensive diagnostic assessment:

    Query: {query}

    Corpus:
    {corpus}

    Please synthesize the information to provide an informative and contextually appropriate diagnostic assessment. Focus on potential diagnoses related to cardiology, neurology, and gastroenterology. Include relevant information about symptoms, risk factors, and suggested diagnostic procedures. Remember to emphasize the importance of professional medical evaluation for a definitive diagnosis.Don't give any disclaimer please.Your are an Intelligent Medical Assistant for Clinical Queries.
    """
    return llm.predict(prompt)

### Main-Function ####
def main():
    crew = setup_medical_qa_system()
    
    # Example usage with text query and lab report
    text_query = "A 55-year-old man experiences severe chest pain and shortness of breath."
    lab_report = "Blood pressure: 160/95 mmHg, Heart rate: 110 bpm, Troponin I: 0.5 ng/mL"

    query = MedicalQuery(text=text_query, lab_report=lab_report)
    result = crew.kickoff(query)

    print("\n############# Final Diagnostic Assessment #############")
    print(result.final_answer)

if __name__ == "__main__":
    main()