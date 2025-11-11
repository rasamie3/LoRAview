from datetime import datetime


class InterviewFunctions:
    def __init__(self, chat_pipeline, tokenizer):
        self.chat_pipeline = chat_pipeline
        self.tokenizer = tokenizer


    def generate_question(self, practice=False):
        """
        Generates a question using k-shot (few-shot) prompting.
        We provide k=2 examples to show the model the desired Q&A pattern.
        """
        
        system_msg = "You are an experienced ML interviewer. Follow the pattern of the examples to generate one single, challenging machine learning interview question. Do not answer the question."
        
        # Define the k-shot examples (k=2)
        # We show the model a fake conversation
        examples = [
            {
                "role": "user", 
                "content": "Give me a standard interview question."
            },
            {
                "role": "assistant",
                "content": "Can you explain the difference between L1 and L2 regularization and their respective effects on model weights?"
            },
            {
                "role": "user",
                "content": "Give me a more advanced one."
            },
            {
                "role": "assistant",
                "content": "Describe the architecture of a Transformer model. What is the self-attention mechanism, and why is it significant compared to older sequence-to-sequence models like RNNs?"
            }
        ]
        
        if practice:
            final_user_msg = "Generate a new machine learning question for practice."
        else:
            final_user_msg = "Ask one new, challenging machine learning interview question."
            
        messages = [
            {"role": "system", "content": system_msg}
        ]
        
        messages.extend(examples) 
        
        messages.append({"role": "user", "content": final_user_msg})
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            response = self.chat_pipeline(
                prompt,
                max_new_tokens=70  
            )
            
            result = response[0]["generated_text"].strip()
            return result
            
        except Exception as e:
            print(f"ERROR in generate_question: {e}")
            return "Error generating question"
    
    def answer_question(self, user_question):
        prompt = f"Answer this machine learning question clearly and concisely:\n\nQuestion: {user_question}\n\nAnswer:"
        
        try:
            response = self.chat_pipeline(prompt, max_new_tokens=512, temperature=0.7)
            return response[0]["generated_text"].strip()
        except Exception as e:
            print(f"ERROR in answer_question: {e}")
            return "Error generating answer"
    
    def review_answer(self, user_question, user_answer):
        prompt = f"""Review this ML interview answer:
        Question: {user_question}
        Candidate Answer: {user_answer}
        Rate the Cadidate answer from 0 to 10 and correct it if it's wrong:
        Review:"""
        
        try:
            response = self.chat_pipeline(prompt, max_new_tokens=512, temperature=0.7)
            return response[0]["generated_text"].strip()
        except Exception as e:
            print(f"ERROR in review_answer: {e}")
            return "Error generating review"
    
    def generate_question_batch(self, num_questions=5):
        questions = []
        for i in range(num_questions):
            print(f"Generating question {i+1}/{num_questions}...")
            question = self.generate_question()
            questions.append({
                "id": i + 1,
                "question": question,
                "timestamp": datetime.now().isoformat()
            })
        return questions
    
    def debug_session(self):
        print("=== Debugging Session ===")
        
        print("\n1. Testing basic text generation...")
        basic_result = self.test_basic_generation()
        
        print("\n2. Testing question generation (method 1)...")
        question1 = self.generate_question(practice=True)
        
        print("\n3. Testing question generation (method 2)...")
        question2 = self.generate_question_alternative(practice=True)
        
        print("\n4. Testing with different prompts...")
        test_prompts = [
            "What is supervised learning?",
            "Question: What is the difference between bias and variance?",
            "ML Interview Question:",
            "Ask me about machine learning:"
        ]
        
        for i, test_prompt in enumerate(test_prompts, 1):
            print(f"\n  Test {i}: '{test_prompt}'")
            try:
                result = self.chat_pipeline(test_prompt, max_new_tokens=100, temperature=0.7)
                print(f"  Result: '{result[0]['generated_text'][:200]}...'")
            except Exception as e:
                print(f"  Error: {e}")
    
    def interactive_qa_session(self):
        print("=== Interactive Q&A Session ===")
        print("Type 'quit' to exit, 'debug' for debugging")
        
        while True:
            user_input = input("\nEnter command or ML question: ")
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'debug':
                self.debug_session()
                continue
            
            print("\nGenerating answer...")
            answer = self.answer_question(user_input)
            print(f"\nAnswer: {answer}")
    
    def practice_session(self):
        print("=== Practice Session ===")
        print("Type 'debug' for debugging, 'quit' to exit")
        
        while True:
            choice = input("\nGenerate question? (y/n/debug/quit): ")
            if choice.lower() in ['n', 'quit']:
                break
            elif choice.lower() == 'debug':
                self.debug_session()
                continue
            elif choice.lower() == 'y':
                print("\nGenerating question...")
                question = self.generate_question(practice=True)
                print(f"\nQuestion: {question}")
                
                see_answer = input("\nSee model's answer? (y/n): ")
                if see_answer.lower() == 'y':
                    answer = self.answer_question(question)
                    print(f"\nModel's Answer: {answer}")