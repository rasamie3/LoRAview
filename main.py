"""
ML Interview Assistant - Main Application
A modular system for ML interview question generation, answering, and review.
"""

from model_loader import ModelLoader
from interview_functions import InterviewFunctions
from file_utils import FileUtils
from datetime import datetime


class MLInterviewAssistant:
    def __init__(self):
        self.model_loader = None
        self.interview_functions = None
        self.file_utils = FileUtils()
        self.qa_session_data = []
        self.practice_session_data = []
    
    def initialize_model(self, base_model_name=None):
        print("Initializing ML Interview Assistant...")
        if base_model_name:
            self.model_loader = ModelLoader(base_model_name=base_model_name)
        else:
            self.model_loader = ModelLoader()    
        chat_pipeline = self.model_loader.load_model()
        self.interview_functions = InterviewFunctions(chat_pipeline)
        print("Initialization complete!")
    
    def question_generation_mode(self):
        """Mode for generating and saving questions"""
        print("\n=== Question Generation Mode ===")
        
        while True:
            print("\nOptions:")
            print("1. Generate single question")
            print("2. Generate batch of questions")
            print("3. Save current questions to file")
            print("5. Back to main menu")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == "1":
                question = self.interview_functions.generate_question()
                print(f"\nGenerated Question:\n{question}")
                
                save_choice = input("\nSave this question? (y/n): ")
                if save_choice.lower() == 'y':
                    self.file_utils.save_questions_to_md([{"question": question}])
            
            elif choice == "2":
                try:
                    num_questions = int(input("How many questions? (default 5): ") or "5")
                    questions = self.interview_functions.generate_question_batch(num_questions)
                    
                    print(f"\nGenerated {len(questions)} questions:")
                    for i, q in enumerate(questions, 1):
                        print(f"\n{i}. {q['question']}")
                    
                    save_choice = input("\nSave these questions? (y/n): ")
                    if save_choice.lower() == 'y':
                        md_file = self.file_utils.save_questions_to_md(questions)
                        pdf_choice = input("Also save as PDF? (y/n): ")
                        if pdf_choice.lower() == 'y':
                            self.file_utils.save_to_pdf(md_file)
                
                except ValueError:
                    print("Please enter a valid number.")
            
            elif choice == "3":
                if not self.qa_session_data:
                    print("No questions to save. Generate some questions first.")
                else:
                    md_file = self.file_utils.save_questions_to_md(self.qa_session_data)
                    pdf_choice = input("Also save as PDF? (y/n): ")
                    if pdf_choice.lower() == 'y':
                        self.file_utils.save_to_pdf(md_file)
            
            elif choice == "5":
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    def answer_review_mode(self):
        """Mode for answering questions and getting reviews"""
        print("\n=== Answer Review Mode ===")
        
        while True:
            print("\nOptions:")
            print("1. Ask a question and get answer")
            print("2. Review your answer to a question")
            print("3. Interactive Q&A session")
            print("4. Save Q&A session")
            print("5. Back to main menu")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == "1":
                question = input("Enter your ML question: ")
                print("\nGenerating answer...")
                answer = self.interview_functions.answer_question(question)
                print(f"\nAnswer: {answer}")
                
                # Save to session data
                self.qa_session_data.append({
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                })
            
            elif choice == "2":
                question = input("Enter the question: ")
                user_answer = input("Enter your answer: ")
                print("\nGenerating review...")
                review = self.interview_functions.review_answer(question, user_answer)
                print(f"\nReview: {review}")
                
                # Save to session data
                self.qa_session_data.append({
                    "question": question,
                    "answer": user_answer,
                    "review": review,
                    "timestamp": datetime.now().isoformat()
                })
            
            elif choice == "3":
                self.interview_functions.interactive_qa_session()
            
            elif choice == "4":
                if not self.qa_session_data:
                    print("No Q&A session data to save.")
                else:
                    md_file = self.file_utils.save_qa_session_to_md(self.qa_session_data)
                    pdf_choice = input("Also save as PDF? (y/n): ")
                    if pdf_choice.lower() == 'y':
                        self.file_utils.save_to_pdf(md_file)
            
            elif choice == "5":
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    def practice_mode(self):
        """Mode for practice sessions with generated questions"""
        print("\n=== Practice Mode ===")
        
        while True:
            print("\nOptions:")
            print("1. Start practice session")
            print("2. Save practice session")
            print("3. Back to main menu")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == "1":
                self.interview_functions.practice_session()
            
            elif choice == "2":
                if not self.practice_session_data:
                    print("No practice session data to save.")
                else:
                    md_file = self.file_utils.save_practice_session(self.practice_session_data)
                    pdf_choice = input("Also save as PDF? (y/n): ")
                    if pdf_choice.lower() == 'y':
                        self.file_utils.save_to_pdf(md_file)
            
            elif choice == "3":
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    def run(self):
        """Main application loop"""
        print("Welcome to ML Interview Assistant!")
        print("Loading model...")
        
        try:
            self.initialize_model()
        except Exception as e:
            print(f"Error initializing model: {e}")
            print("Make sure the model files are in the correct location.")
            return
        
        while True:
            print("\n" + "="*50)
            print("ML INTERVIEW ASSISTANT")
            print("Default Base Model is : ", self.model_loader.base_model_name)

            print("="*50)
            print("0. Change Base Model")
            print("1. Question Generation Mode")
            print("2. Answer Review Mode")
            print("3. Practice Mode")
            print("4. Exit")
            
            choice = input("\nSelect mode (1-4): ")
            
            if choice == "0":
                print("Choose Base Model")
                print("1. TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                print("2. Qwen/Qwen2.5-0.5B-Instruct")
                
                model_choice = input("\nSelect Base Model (1-2): ")
                if model_choice == "1":
                    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                elif model_choice == "2":
                    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
                else:
                    print("Invalid choice. Choosing default model: ", self.model_loader.base_model_name)    
                try:
                    self.initialize_model(base_model_name = base_model)
                except Exception as e:
                    print(f"Error initializing model: {e}")
                    print("Make sure the model files are in the correct location.")
                    return
                
            elif choice == "1":
                self.question_generation_mode()
            elif choice == "2":
                self.answer_review_mode()
            elif choice == "3":
                self.practice_mode()
            elif choice == "4":
                print("Thank you for using ML Interview Assistant!")
                break
            else:
                print("Invalid choice. Please try again.")


def main():
    """Main entry point"""
    assistant = MLInterviewAssistant()
    assistant.run()


if __name__ == "__main__":
    main() 