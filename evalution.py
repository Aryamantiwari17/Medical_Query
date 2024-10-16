from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from typing import List, Tuple
import json
import nltk
from less import setup_medical_qa_system, MedicalQuery
###Code for evalution Purpose######

nltk.download('punkt', quiet=True)

def calculate_bleu(reference: str, candidate: str) -> float:
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)

def calculate_rouge(reference: str, candidate: str) -> float:
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    return scores['rouge-1']['f']

def medical_accuracy_score(reference: str, candidate: str) -> float:
    # This is a placeholder function. In a real-world scenario, you would implement
    # a more sophisticated method to evaluate medical accuracy.
    keywords = set(reference.lower().split()) & set(candidate.lower().split())
    return len(keywords) / len(set(reference.lower().split()))

def evaluate_response(reference: str, candidate: str) -> Tuple[float, float, float]:
    bleu = calculate_bleu(reference, candidate)
    rouge = calculate_rouge(reference, candidate)
    accuracy = medical_accuracy_score(reference, candidate)
    return bleu, rouge, accuracy

def construct_reference_answer(possible_diagnoses):
    answer = "Based on the symptoms, possible diagnoses include:\n"
    for diagnosis in possible_diagnoses:
        answer += f"- {diagnosis['diagnosis']}: {diagnosis['description']}\n"
        if diagnosis['risk_factors']:
            answer += f"  Risk factors: {', '.join(diagnosis['risk_factors'])}\n"
        if diagnosis['diagnostic_tests']:
            answer += f"  Diagnostic tests: {', '.join(diagnosis['diagnostic_tests'])}\n"
    return answer.strip()

def main_evaluation():
    with open('test_dataset.json', 'r') as file:
        test_data = json.load(file)
    
    # Setup the medical QA system (assuming this function exists)
    crew = setup_medical_qa_system()
    
    total_bleu, total_rouge, total_accuracy = 0, 0, 0
    count = 0
    
    for item in test_data:
        if 'question' not in item or 'possible_diagnoses' not in item:
            print(f"Skipping item due to missing keys: {item}")
            continue

        query = MedicalQuery(text=item['question'], lab_report='')
        response = crew.kickoff(query)
        
        generated_answer = getattr(response, 'final_answer', 'No answer generated')
        reference_answer = construct_reference_answer(item['possible_diagnoses'])
        
        bleu, rouge, accuracy = evaluate_response(reference_answer, generated_answer)
        
        total_bleu += bleu
        total_rouge += rouge
        total_accuracy += accuracy
        count += 1
    
    if count > 0:
        avg_bleu = total_bleu / count
        avg_rouge = total_rouge / count
        avg_accuracy = total_accuracy / count
        
        print(f"Average BLEU score: {avg_bleu:.4f}")
        print(f"Average ROUGE-1 score: {avg_rouge:.4f}")
        print(f"Average Medical Accuracy score: {avg_accuracy:.4f}")
    else:
        print("No items were processed. Check your dataset and error messages.")

if __name__ == "__main__":
    main_evaluation()