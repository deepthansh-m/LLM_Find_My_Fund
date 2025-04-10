# LLM, Find My Fund!

## Overview

"LLM, Find My Fund!" is a project designed to accurately match Indian fund/security queries to the correct financial product in a dataset of mutual funds, ETFs, and other securities. The project leverages a small language model (using Sentence Transformers) and FAISS for fast vector similarity search and efficient retrieval. It is built with a focus on accuracy, explainability, and a small footprint for deployment on commodity hardware.

**Key Features:**
- **High Accuracy:** Matches ambiguous or incomplete queries by incorporating fund metadata.
- **Efficient Vector Search:** Utilizes FAISS to achieve fast, real-time retrieval.
- **Small Model Footprint:** Uses lightweight models like DistilBERT or TinyLlama for embeddings.
- **Easy Extensibility:** Separate training and prediction scripts let you quickly retrain the index once your dataset is available.
- **Frontend & Backend:** A FastAPI backend exposes `/train` and `/predict` endpoints, while a simple HTML/JavaScript frontend lets users query the model.

## Directory Structure

```
LLM_Find_My_Fund/
├── backend/
│   ├── main.py       # FastAPI entry point with /train and /predict endpoints
│   ├── train.py      # Training script to build the FAISS index from your dataset
│   ├── predict.py    # Query processing: returns the best matching fund/security
│   └── utils.py      # Utility functions for saving/loading the FAISS index and metadata
├── frontend/
│   ├── index.html    # HTML page for the UI
│   ├── script.js     # JavaScript code to communicate with the backend API
│   └── style.css     # Basic styling for the frontend
├── requirements.txt  # Python dependencies needed for the project
└── README.md         # This file
```

## Setup Instructions

### 1. Clone or Create the Project in PyCharm

- **Clone the Repository:**  
  If you have a repository URL, clone it using:
  ```bash
  git clone <your-repo-url>
  ```
  Otherwise, create a new project in PyCharm and add the directory structure as shown above.

- **Set Up a Virtual Environment:**
  PyCharm automatically creates a virtual environment if you select that option on project creation. Alternatively, you can create one manually:
  ```bash
  python -m venv env
  source env/bin/activate         # On Windows: env\Scripts\activate
  ```

### 2. Install Dependencies
Open the PyCharm Terminal (or your system terminal) and run:

```bash
  pip install -r requirements.txt
```
This installs packages such as FastAPI, uvicorn, pandas, faiss-cpu, sentence-transformers, and others.

### 3. Configuring PyCharm
#### A. Create Run/Debug Configurations for the Backend

**FastAPI Server Configuration:**
- Go to Run > Edit Configurations.
- Click the "+" icon and choose Python.
- Set the Script path to backend/main.py.
- Optionally, add parameters or environment variables if needed.
- Save this configuration to run your FastAPI backend.

**Training Script Configuration:**
- Create another Python configuration in Run > Edit Configurations.
- Set the Script path to backend/train.py.
- In the "Parameters" field, add your dataset path. For example:
  ```
  --dataset_path data/indian_funds.csv
  ```
- Save the configuration to run your training script when your dataset is available.

### 4. Running the Project
#### A. Train the Model

Once your dataset (e.g., data/indian_funds.csv) is available, run the training script from PyCharm or the terminal:

```bash
  python backend/train.py --dataset_path data/indian_funds.csv
```
This script will:

- Load your CSV dataset (which must include at least a fund_name column, with optional metadata like category and fund_house).
- Compute embeddings using the Sentence Transformer model.
- Build a FAISS index and save it along with the associated metadata.

#### B. Start the Backend Server

Run the FastAPI server using the PyCharm run configuration or from the terminal:

```bash
  uvicorn backend.main:app --reload
```
The backend API will be accessible on http://localhost:8000.

#### C. Use the Frontend

- Open the frontend/index.html file in your browser.
- In PyCharm, right-click on the file and select "Open in Browser".
- Use the UI to enter a fund query (e.g., "icici infra") and click Search.
- The query will be sent to your backend's /predict endpoint, and the matched fund will be displayed.

## API Endpoints

### /train
- **Method:** POST
- **Description:** Triggers the training process to build or update the FAISS index from your dataset.
- **Request JSON:**
  ```json
  {
      "dataset_path": "path/to/your/dataset.csv"
  }
  ```
- **Response:** Confirmation message upon successful training.

### /predict
- **Method:** POST
- **Description:** Accepts a query and returns the best matching fund/security.
- **Request JSON:**
  ```json
  {
      "query": "icici infra"
  }
  ```
- **Response JSON:**
  ```json
  {
      "matched_fund": "The matched fund name here"
  }
  ```

## Customization & Future Enhancements

- **Model Fine-Tuning:**
  Integrate advanced fine-tuning methods (like LoRA or PEFT) if you require even better accuracy for your specific dataset.
- **Metadata Utilization:**
  Modify how the metadata is combined with the fund name during embedding calculation to further boost matching performance.
- **UI Enhancements:**
  Expand the frontend for additional filtering or explanation of the matching logic.
- **Deployment:**
  Package the project using Docker or deploy using cloud-based services if scaling or production-readiness is required.

## Final Notes

- Ensure your dataset file is correctly formatted and available in the specified path before training.
- Use Git integration within PyCharm for version control as you extend the project.
- Utilize breakpoints and PyCharm's debugger to troubleshoot and optimize your code.

Happy coding and good luck