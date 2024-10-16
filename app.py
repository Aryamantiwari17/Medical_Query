# app.py
from flask import Flask, render_template, request, flash
from less import setup_medical_qa_system, MedicalQuery
import re
##UI for Medical Clinical Query
app = Flask(__name__)


# Initialize the medical QA system
crew = setup_medical_qa_system()



def format_assessment(assessment):
    # Handle bold text (marked with **)
    assessment = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', assessment)
    
    # Handle bullet points (marked with *)
    assessment = re.sub(r'(?<=\n)\*\s+', r'<li>', assessment)  # Start a list item on new line
    assessment = re.sub(r'(?<!<li>)(?=\n)(?=\* )', r'<ul>\n', assessment)  # Start unordered list
    assessment = re.sub(r'(?<=</li>)\n(?!<ul>)', r'</ul>\n', assessment)  # Close unordered list
    assessment = re.sub(r'(?<!<li>)(?=\n)', r'<br>', assessment)  # Convert line breaks

    # Wrap in paragraphs
    assessment = re.sub(r'\n\n+', r'</p><p>', assessment)  # Paragraph breaks
    assessment = '<p>' + assessment.strip() + '</p>'  # Wrap everything in a paragraph
    
    return assessment


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_query = request.form['text_query']
        lab_report = request.form.get('lab_report', '')  # Use get() with a default value
        
        try:
            query = MedicalQuery(text=text_query, lab_report=lab_report)
            result = crew.kickoff(query)
            formatted_assessment = format_assessment(result.final_answer)
            
            return render_template('result.html', 
                                   text_query=text_query,
                                   lab_report=lab_report,
                                   final_assessment=formatted_assessment)
        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'error')
            return render_template('index.html')
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)