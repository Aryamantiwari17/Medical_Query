# Medical Diagnostic Query System

This project is a Flask-based web application that uses AI to provide medical assessments based on user queries and lab reports. It integrates various AI models and tools, including Wikipedia search, a custom medical database, and PMC Open Access articles.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* Python 3.7 or higher
* pip (Python package manager)
* Git

## Installation

Follow these steps to set up the project on your local machine:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/medical-clinical-query-system.git
   cd medical-clinical-query-system
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   * On Windows:
     ```
     venv\Scripts\activate
     ```
   * On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Set up the Google API key:
   * Create a `.env` file in the project root directory
   * Add your Google API key to the file:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

6. Download NLTK data:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

7. Load the custom medical dataset:
   * Ensure you have a `dev.json` file in the project root directory containing your medical QA data
   * The file should have the following structure:
     ```json
     {
       "data": [
         {
           "question": "Medical question here",
           "answer": "Corresponding answer here"
         },
         ...
       ]
     }
     ```

## Running the Application

To run the Medical Clinical Query System:

1. Ensure you're in the project directory with the virtual environment activated.

2. Start the Flask application:
   ```
   python app.py
   ```

3. Open a web browser and navigate to `http://localhost:5001`

4. Use the web interface to submit medical queries and lab reports.

## Usage

1. Enter your medical query in the text box provided.
2. If you have lab report data, enter it in the designated field.
3. Click the "Submit" button to get a medical assessment.
4. The system will process your query using various AI tools and provide a comprehensive assessment.

## Evalution

1. Run the evalution.py indivually to get the evalute score based on the model.

## Troubleshooting

* If you encounter any issues with missing modules, ensure all dependencies are installed:
  ```
  pip install -r requirements.txt
  ```
* If the FAISS index fails to load, delete the `faiss_data.pkl` file and run the application again to rebuild the index.
* For any Google API-related issues, verify that your API key is correct and has the necessary permissions.

## Output PIC
![output](https://github.com/user-attachments/assets/1adc978d-6490-448b-8ba6-a2703057e349)
![UI](https://github.com/user-attachments/assets/ec578053-e636-4130-8ba9-e519d1ec9e56)


## Contributing

Contributions to the Medical Clinical Query System are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).
