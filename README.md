# ResumeExtractorAndScorer

## Overview

**Key\_Information\_From\_Resume\_Extractor\_and\_Scorer** is a Python-based tool designed to extract key details from resumes and automatically score them based on relevance to specific job descriptions. It leverages NLP and ML techniques to parse resumes, identify important information, and provide a structured scoring output for easier candidate evaluation.

## Features

* **Resume Parsing**: Extracts candidate name, contact information, skills, education, and work experience.
* **Skill Matching and Scoring**: Scores resumes based on alignment with desired job skills and qualifications.
* **Batch Processing**: Process multiple resumes at once.
* **Customizable Scoring**: Adjust weight for skills, experience, and education.
* **Output Formats**: Generates structured CSV/JSON output for easy analysis.

## Repository Structure

```
Key_Information_From_Resume_Extractor_and_Scorer/
│
├── backend/
│   ├── main.py           # Entry point for backend processing
│   ├── resume_parser.py  # Handles extraction of resume information
│   ├── scorer.py         # Implements scoring logic
│   └── requirements.txt  # Backend dependencies
│
├── frontend/
│   ├── App.js            # React main component
│   ├── components/       # UI components for uploading and viewing results
│   ├── package.json      # Frontend dependencies
│   └── public/           # Public assets
│
├── resumes_sample/       # Sample resumes for testing
├── job_descriptions/     # Job descriptions to match against
└── README.md
```

## Installation

### Prerequisites

* Python 3.8+
* Node.js and npm
* Optional: Docker for containerized setup

### Backend Setup

1. Navigate to the backend folder:

   ```bash
   cd backend
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend service:

   ```bash
   python main.py
   ```

### Frontend Setup

1. Navigate to the frontend folder:

   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:

   ```bash
   npm install
   ```

3. Start the React app:

   ```bash
   npm start
   ```

   The frontend will be available at `http://localhost:3000`.

### Docker Setup (Optional)

1. Ensure Docker is installed and running.
2. From the project root, run:

   ```bash
   docker-compose up --build
   ```

   This will build and start both backend and frontend services.

## Usage

1. Upload resumes via the frontend UI or place them in the `resumes_sample/` folder.
2. Provide a job description from the `job_descriptions/` folder.
3. Run extraction and scoring.
4. View results in the frontend dashboard or as CSV/JSON output.

## Technologies Used

* **Python 3.8+** – Backend processing
* **NLP** – spaCy, Transformers for extracting and understanding text
* **Machine Learning** – Scikit-learn or similar for scoring and ranking resumes
* **React.js** – Frontend user interface
* **Docker** – Containerization for deployment
* **Pandas** – For structured output and data manipulation

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License.
