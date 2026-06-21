# RFP Intelligence System (RAG)

## 1. Project Overview

The **RFP Intelligence System** is an AI-powered Retrieval-Augmented Generation (RAG) system designed to help proposal teams process Requests for Proposal (RFPs), extract requirements, retrieve reusable company knowledge, and generate grounded proposal drafts with citations.

The system uses a controlled workflow instead of a free-form chatbot. Users can upload RFP documents, review extracted requirements, retrieve proposal evidence, approve evidence, generate a proposal, export the output, and evaluate the result.

---

## 2. Main Features

* Upload and process RFP files.
* Parse PDF, DOCX, PPTX, XLSX, Markdown, and TXT files.
* Clean and normalize extracted document text.
* Split documents into searchable chunks.
* Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
* Store chunks in ChromaDB.
* Use two separate vector databases:

  * `rfp_db` for current/client RFP documents.
  * `proposal_db` for historical proposals and reusable company knowledge.
* Extract structured RFP requirements using OpenAI.
* Retrieve proposal evidence from the company knowledge base.
* Review and approve evidence before generation.
* Generate grounded proposal drafts with citations.
* Export proposal output as JSON, Markdown, and optional PDF.
* Evaluate retrieval and generation quality.

---

## 3. Project Architecture

The project uses two separate ChromaDB databases:

```text
data/chroma/rfp_db
collection: rfp_documents
```

Used for current/client RFP documents.

```text
data/chroma/proposal_db
collection: proposal_knowledge
```

Used for old proposals, company knowledge, project descriptions, and reusable evidence.

This separation prevents the system from treating client requirements as company evidence.

---

## 4. Folder Structure

```text
RFP_Project/
│
├── data/
│   ├── raw/
│   │   ├── rfp_uploads/
│   │   └── proposal_knowledge/
│   │
│   ├── processed/
│   │   ├── file_manifest.csv
│   │   ├── parsed_elements.jsonl
│   │   ├── cleaned_documents.jsonl
│   │   ├── chunks.jsonl
│   │   ├── rfp_context/
│   │   ├── requirements/
│   │   ├── proposal_context/
│   │   └── generated/
│   │
│   ├── chroma/
│   │   ├── rfp_db/
│   │   └── proposal_db/
│   │
│   └── evaluation/
│
├── src/
│   ├── data/
│   │   ├── create_manifest.py
│   │   ├── parse_documents.py
│   │   ├── clean_normalize.py
│   │   ├── create_chunks.py
│   │   └── insert_chroma.py
│   │
│   ├── rag/
│   │   ├── retrieve_rfp_context.py
│   │   ├── extract_requirements.py
│   │   ├── retrieve_proposals.py
│   │   └── generate_proposal.py
│   │
│   ├── rendering/
│   │   └── render_proposal.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_retrieval.py
│   │   └── evaluate_generation.py
│   │
│   └── ui/
│       └── streamlit_app.py
│
├── notebooks/
│   └── rfp_dynamic_evaluation.ipynb
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## 5. Requirements

Install the required Python packages using:

```powershell
pip install -r requirements.txt
```

Main libraries used:

* `streamlit`
* `chromadb`
* `sentence-transformers`
* `openai`
* `pandas`
* `python-dotenv`
* `pypdf`
* `python-docx`
* `python-pptx`
* `openpyxl`
* `markdown`
* `xhtml2pdf`

---

## 6. Environment Setup

### Step 1: Open the project folder

```powershell
cd "D:\Artificial intelligence\SDA\rfp_project_final\RFP_Project"
```

### Step 2: Create a virtual environment

```powershell
python -m venv rfp_env
```

### Step 3: Activate the environment

```powershell
.\rfp_env\Scripts\activate
```

You should see:

```text
(rfp_env)
```

at the beginning of the terminal line.

### Step 4: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 5: Create `.env` file

Create a `.env` file in the root project folder:

```text
RFP_Project/.env
```

Add:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.5
```

Do not push `.env` to GitHub.

---

## 7. Data Setup

Place current RFP documents in:

```text
data/raw/rfp_uploads/
```

Place historical proposal and company knowledge documents in:

```text
data/raw/proposal_knowledge/
```

Example:

```text
data/raw/rfp_uploads/my_opportunity/
data/raw/proposal_knowledge/company_capabilities/
data/raw/proposal_knowledge/past_projects/
```

Supported file types:

```text
.pdf
.docx
.pptx
.xlsx
.md
.txt
```

---

## 8. Run the Full Data Pipeline Manually

Use this process if you want to build the knowledge base from the terminal.

### Step 1: Create manifest

```powershell
python -m src.data.create_manifest
```

This creates:

```text
data/processed/file_manifest.csv
```

### Step 2: Parse documents

```powershell
python -m src.data.parse_documents
```

This creates:

```text
data/processed/parsed_elements.jsonl
```

### Step 3: Clean and normalize text

```powershell
python -m src.data.clean_normalize
```

This creates:

```text
data/processed/cleaned_documents.jsonl
```

### Step 4: Create chunks

```powershell
python -m src.data.create_chunks
```

This creates:

```text
data/processed/chunks.jsonl
```

### Step 5: Insert chunks into ChromaDB

```powershell
python -m src.data.insert_chroma --reset
```

This creates or rebuilds:

```text
data/chroma/rfp_db
data/chroma/proposal_db
```

---

## 9. Run the Streamlit Application

Start the app with:

```powershell
streamlit run src/ui/streamlit_app.py
```

The app will open in your browser.

The Streamlit workflow includes:

1. **Upload & Process**

   * Upload RFP files.
   * Enter opportunity ID.
   * Run RFP processing pipeline.

2. **Review Requirements**

   * Review extracted requirements.
   * Select requirements.
   * Choose proposal type: business, technical, or both.

3. **Review Evidence**

   * Retrieve proposal knowledge from `proposal_db`.
   * Review evidence candidates.
   * Approve evidence.

4. **Generate & Export**

   * Generate proposal.
   * Preview Markdown.
   * Download JSON, Markdown, and optional PDF.

---

## 10. Run RFP Context Retrieval Manually

Example:

```powershell
python -m src.rag.retrieve_rfp_context --opportunity-id full_cycle_with_max_queries_testing --top-k 8 --output data/processed/rfp_context/full_cycle_with_max_queries_testing_rfp_context.json --max-preview 20
```

This retrieves relevant chunks from:

```text
rfp_db / rfp_documents
```

---

## 11. Run Requirement Extraction Manually

Example:

```powershell
python -m src.rag.extract_requirements --input data/processed/rfp_context/full_cycle_with_max_queries_testing_rfp_context.json --output data/processed/requirements/full_cycle_with_max_queries_testing_extracted_requirements.json
```

This uses OpenAI to extract structured requirements from retrieved RFP context.

---

## 12. Run Proposal Evidence Retrieval Manually

Example:

```powershell
python -m src.rag.retrieve_proposals --requirements data/processed/requirements/full_cycle_with_max_queries_testing_extracted_requirements.json --proposal-type both --top-k 5 --output data/processed/proposal_context/full_cycle_with_max_queries_testing_both_proposal_context.json
```

This retrieves reusable proposal evidence from:

```text
proposal_db / proposal_knowledge
```

---

## 13. Generate Proposal Manually

Example:

```powershell
python -m src.rag.generate_proposal --requirements data/processed/requirements/full_cycle_with_max_queries_testing_extracted_requirements.json --proposal-context data/processed/proposal_context/full_cycle_with_max_queries_testing_both_proposal_context.json --proposal-type both --output data/processed/generated/full_cycle_with_max_queries_testing_both_proposal.json
```

The generated proposal is saved as structured JSON.

---

## 14. Render Proposal to Markdown

Example:

```powershell
python -m src.rendering.render_proposal --input data/processed/generated/full_cycle_with_max_queries_testing_both_proposal.json --output data/processed/generated/full_cycle_with_max_queries_testing_both_proposal.md
```

The Markdown output can be opened and reviewed directly.

---

## 15. Evaluation

The project includes two evaluation types:

1. Retrieval evaluation
2. Generation evaluation

---

## 15.1 Retrieval Evaluation

Retrieval evaluation checks whether the system retrieves expected chunks from ChromaDB.

Run:

```powershell
python src/evaluation/evaluate_retrieval.py --golden data/evaluation/golden_requirements_v2.json --top-k 5 --results-csv data/evaluation/retrieval_results_latest.csv --summary-json data/evaluation/retrieval_summary_latest.json
```

Print summary:

```powershell
Get-Content data/evaluation/retrieval_summary_latest.json
```

Latest confirmed retrieval evaluation result:

```text
Top-k: 5
Tests: 5

Overall:
Mean Precision@K: 0.24
Mean Recall@K: 0.80
Hit Rate@K: 0.80
Mean Reciprocal Rank: 0.80
Top-1 Chunk Accuracy: 0.80
Source Accuracy@K: 0.80
Page Accuracy@K: 1.00
Source + Page Accuracy@K: 1.00
```

RFP retrieval performed strongly:

```text
RFP tests: 3
Recall@K: 1.0
Hit Rate@K: 1.0
MRR: 1.0
Top-1 Accuracy: 1.0
Source Accuracy@K: 1.0
Page Accuracy@K: 1.0
```

Proposal retrieval used only 2 tests, so the result is a small validation sample, not a full benchmark.

---

## 15.2 Generation Evaluation

Generation evaluation checks whether the final proposal is complete, grounded, cited correctly, and professionally useful.

Run:

```powershell
python src/evaluation/evaluate_generation.py --proposal data/processed/generated/full_cycle_with_max_queries_testing_both_proposal.json --requirements data/processed/requirements/full_cycle_with_max_queries_testing_extracted_requirements.json --proposal-context data/processed/proposal_context/full_cycle_with_max_queries_testing_both_proposal_context.json --output data/evaluation/generation_results_full_cycle_with_max_queries_testing.json --summary-csv data/evaluation/generation_summary_full_cycle_with_max_queries_testing.csv --human-review data/evaluation/human_generation_review_full_cycle_with_max_queries_testing.csv --judge-model gpt-5.5
```

If OpenAI quota is unavailable, run deterministic-only evaluation:

```powershell
python src/evaluation/evaluate_generation.py --proposal data/processed/generated/full_cycle_with_max_queries_testing_both_proposal.json --requirements data/processed/requirements/full_cycle_with_max_queries_testing_extracted_requirements.json --proposal-context data/processed/proposal_context/full_cycle_with_max_queries_testing_both_proposal_context.json --output data/evaluation/generation_results_full_cycle_with_max_queries_testing.json --summary-csv data/evaluation/generation_summary_full_cycle_with_max_queries_testing.csv --human-review data/evaluation/human_generation_review_full_cycle_with_max_queries_testing.csv --judge-model none
```

Latest confirmed generation evaluation result:

```text
Requirements: 94
Compliance rows: 94
Proposal citations: 12
Proposal context chunks: 53
Proposal knowledge source files: 8
```

Deterministic metrics:

```text
Requirement coverage: 1.0
Mandatory requirement coverage: 1.0
Narrative section completion: 1.0
Citation ID accuracy: 1.0
Citation metadata accuracy: 1.0
Compliance evidence coverage: 0.2021
```

LLM-as-judge metrics:

```text
Faithfulness: 5 / 5
Answer relevance: 4 / 5
Requirement alignment: 3 / 5
Professional quality: 4 / 5
Citation support rate: 1.0
```

---

## 16. Common Issues and Fixes

### Issue 1: OpenAI quota error

If you see an OpenAI quota error, check:

* API key
* billing
* quota
* model name in `.env`

For evaluation, use:

```powershell
--judge-model none
```

to skip LLM-as-judge.

---

### Issue 2: ChromaDB compaction error

If you see:

```text
InternalError: Error in compaction: Failed to apply logs to the metadata segment
```

stop Streamlit and rebuild ChromaDB:

```powershell
Remove-Item -Recurse -Force data\chroma\rfp_db
Remove-Item -Recurse -Force data\chroma\proposal_db
python -m src.data.insert_chroma --reset
```

This is safe because ChromaDB is only the vector index. The source of truth is in `data/raw` and `data/processed`.

---

### Issue 3: C drive storage filling up

Safe cleanup:

```powershell
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
pip cache purge
Remove-Item -Recurse -Force "$env:TEMP\*" -ErrorAction SilentlyContinue
```

To move model cache to D drive:

```powershell
setx HF_HOME "D:\AI_Cache\huggingface"
setx SENTENCE_TRANSFORMERS_HOME "D:\AI_Cache\huggingface"
setx TORCH_HOME "D:\AI_Cache\torch"
```

Restart VS Code or terminal after setting these variables.

---

### Issue 4: Retrieval evaluation looks low

The current retrieval golden dataset is small:

```text
5 total tests
3 RFP tests
2 proposal tests
```

So retrieval evaluation is a validation sample, not a full benchmark.

To improve it, add more golden test cases to:

```text
data/evaluation/golden_requirements_v2.json
```

Recommended target:

```text
10 RFP retrieval tests
10 proposal retrieval tests
```

---

## 17. Important Safety Notes

Do not push these to GitHub:

```text
.env
rfp_env/
data/raw/
data/chroma/
private proposal files
private RFP files
confidential generated proposals
API keys
```

Safe files to push:

```text
src/
requirements.txt
README.md
.gitignore
documentation/
notebooks/ if cleaned
sample/anonymized data only
```

---

## 18. Final Project Result

The final system successfully completed the full RAG workflow:

```text
RFP upload
→ document processing
→ requirement extraction
→ requirement review
→ proposal evidence retrieval
→ evidence review
→ grounded proposal generation
→ citation validation
→ Markdown/PDF export
→ evaluation
```

The latest full-cycle run achieved:

```text
94/94 requirements covered
100% mandatory requirement coverage
100% citation ID accuracy
100% citation metadata accuracy
Faithfulness: 5/5
Citation support rate: 1.0
```

This shows that the system is successful as an SDA final project MVP and demo.

---

## 19. Future Improvements

Recommended future improvements:

* Expand the proposal knowledge base.
* Add category-aware evidence handling.
* Separate evidence-needed requirements from confirmation-needed requirements.
* Improve missing evidence wording.
* Add more golden retrieval test cases.
* Add reranking or hybrid retrieval.
* Improve PDF/DOCX export formatting.
* Add user approval workflow.
* Add authentication and access control for production use.
* Add audit logging and reviewer comments.

```
```
