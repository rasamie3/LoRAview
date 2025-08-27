from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel


base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
adapter_path = "./tinyllama-lora-ftuned-adapted"        

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map=None, torch_dtype="float32")

model = PeftModel.from_pretrained(model, adapter_path)

model = model.merge_and_unload()


chat_pipeline = pipeline(
    'text-generation',
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 512,
    temperature = 0.7,
)

def ask_question():
    prompt = """You are an ML interviewer. Generate one challenging machine learning interview question."""
    return chat_pipeline(prompt)[0]["generated_text"]

def answer_question(user_question):
    prompt = f"""You are an ML expert. Answer the following question clearly and concisely:

                Question: {user_question}
                Answer:"""
    return chat_pipeline(prompt)[0]["generated_text"]


def review_answer(user_question, user_answer):
    prompt = f"""You are an ML interviewer. Review the following candidate's answer to the question.
                Question: {user_question}
                Candidate Answer: {user_answer}

                Provide constructive feedback on correctness, completeness, and clarity."""
    return chat_pipeline(prompt)[0]["generated_text"]

while True:
    mode = input("Choose mode (ask / answer / review / quit): ")

    if mode == "quit":
        break
    elif mode == "ask":
        print(ask_question())
    elif mode == "answer":
        q = input("Enter your ML question: ")
        print(answer_question(q))
    elif mode == "review":
        q = input("Enter the question: ")
        a = input("Enter the candidate answer: ")
        print(review_answer(q, a))