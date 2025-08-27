from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel



class ModelLoader:
    def __init__(self, base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", adapter_path="./tinyllama-lora-ftuned-adapted-v2/"):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.chat_pipeline = None

        # Ensure files are correct
        if self.base_model_name == "Qwen/Qwen2.5-0.5B-Instruct":
            self.adapter_path = "./qwen-lora-ftuned-adapted/"
        elif self.base_model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
            self.adapter_path = "./tinyllama-lora-ftuned-adapted-v2/"
        else: 
            print("Error select the model")
    
    def load_model(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, 
            device_map="auto", 
            torch_dtype="float16", 
            trust_remote_code=True
        )
        
        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        
        print("Merging and unloading adapter...")
        self.model = self.model.merge_and_unload()
    
        print("Creating chat pipeline...")
        self.chat_pipeline = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,  
            temperature=0.7,     
            do_sample=True,      
            top_p=0.9,          
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        print("Model loaded successfully!")
        return self.chat_pipeline
    
    def get_pipeline(self):
        if self.chat_pipeline is None:
            return self.load_model()
        return self.chat_pipeline