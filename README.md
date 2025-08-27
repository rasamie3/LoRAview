# LoRAview - ML Interview Assistant

A command-line machine learning interview assistant built with a LoRA fine-tuned TinyLlama/Qwen model. This terminal-based app helps users practice ML interview questions by generating prompts, providing detailed answers, and reviewing candidate responses, making it an easy and effective way to prepare for interviews directly from the CLI. The model is intentionally not highly accurate, since it was created for learning purposes and trained with limited computational resources, but it works fine for its intended use.

## Project Structure

```
LoRAview/
├── model_loader.py          # Model loading and initialization
├── interview_functions.py   # Core interview functionality
├── file_utils.py           # File operations (MD, PDF)
├── main.py                 # Main application interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── interviewer.py         # Original single-file version
├── LoRA_trainer.ipynb       # Training notebook
├── qwen-lora-ftuned-adapted/       # Fine-tuned model files (qwen)
└── tinyllama-lora-ftuned-adapted-v2/  # Fine-tuned model files (tinyllama)
```

## Features

### 1. Question Generation Mode
- Generate single ML interview questions
- Generate batches of questions
- Save questions to Markdown and PDF formats

### 2. Answer Review Mode
- Ask questions and get AI-generated answers
- Submit your answers for review and feedback
- Interactive Q&A sessions
- Save Q&A sessions to files

### 3. Practice Mode
- Practice with generated questions
- Compare your answers with model answers
- Get feedback on your responses

## Future Development

### Web Version
A web-based version of LoRAview is currently under development.

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present:**
   Make sure the `tinyllama-lora-ftuned-adapted-v2/`or `qwen-lora-ftuned-adapted/` directory contains all the fine-tuned model files.

## Usage

### Running the Application

```bash
python main.py
```

The application will load the model and present you with a menu of options:

```
==================================================
ML INTERVIEW ASSISTANT
==================================================
0. Change Base Model
1. Question Generation Mode
2. Answer Review Mode
3. Practice Mode
4. Exit
```

### Question Generation Mode

This mode allows you to:
- Generate single questions
- Generate batches of questions
- Save questions to Markdown and PDF files

### Answer Review Mode

This mode provides:
- Question answering functionality
- Answer review and feedback
- Interactive Q&A sessions
- Session saving capabilities

### Practice Mode

This mode offers:
- Practice sessions with generated questions
- Answer comparison with model responses
- Detailed feedback on your answers
- Progress tracking

## File Output

The system creates an `output/` directory and saves files with timestamps:

- **Markdown files**: `ml_questions_YYYYMMDD_HHMMSS.md`
- **PDF files**: `ml_questions_YYYYMMDD_HHMMSS.pdf`

## Model Information

- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 or 'Qwen/Qwen2.5-0.5B-Instruct'
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Training**: Custom ML interview dataset
- **Hardware**: CPU-compatible (no GPU required)


### Customization

- **Model Path**: Change `adapter_path` in `model_loader.py`
- **Output Directory**: Modify `output_dir` in `file_utils.py`
- **Generation Parameters**: Adjust `max_new_tokens` and `temperature` in `model_loader.py` (important to enhance model responses sometimes)

### Dependencies

If you encounter import errors, install missing packages:
```bash
pip install transformers torch peft accelerate datasets bitsandbytes
```

For PDF generation:
```bash
pip install markdown2 weasyprint
```