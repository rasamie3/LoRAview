import os
from datetime import datetime
from pathlib import Path


class FileUtils:
    def __init__(self, output_dir="./output"):
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def save_questions_to_md(self, questions, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_questions_{timestamp}.md"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Machine Learning Interview Questions\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, q in enumerate(questions, 1):
                f.write(f"## Question {i}\n\n")
                f.write(f"{q['question']}\n\n")
                
                if 'category' in q and q['category']:
                    f.write(f"**Category:** {q['category']}\n\n")
                if 'timestamp' in q:
                    f.write(f"**Generated:** {q['timestamp']}\n\n")
                
                f.write("---\n\n")
        
        print(f"Questions saved to: {filepath}")
        return filepath
    
    def save_qa_session_to_md(self, qa_pairs, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qa_session_{timestamp}.md"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Q&A Session\n\n")
            f.write(f"Session date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, qa in enumerate(qa_pairs, 1):
                f.write(f"## Q&A {i}\n\n")
                f.write(f"**Question:** {qa['question']}\n\n")
                f.write(f"**Answer:** {qa['answer']}\n\n")
                
                if 'review' in qa and qa['review']:
                    f.write(f"**Review:** {qa['review']}\n\n")
                
                f.write("---\n\n")
        
        print(f"Q&A session saved to: {filepath}")
        return filepath
    
    def save_practice_session(self, practice_data, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"practice_session_{timestamp}.md"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Practice Session Report\n\n")
            f.write(f"Session date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, practice in enumerate(practice_data, 1):
                f.write(f"## Practice Question {i}\n\n")
                f.write(f"**Question:** {practice['question']}\n\n")
                f.write(f"**Model Answer:** {practice['model_answer']}\n\n")
                
                if 'user_answer' in practice and practice['user_answer']:
                    f.write(f"**Your Answer:** {practice['user_answer']}\n\n")
                
                if 'review' in practice and practice['review']:
                    f.write(f"**Review:** {practice['review']}\n\n")
                
                f.write("---\n\n")
        
        print(f"Practice session saved to: {filepath}")
        return filepath
    
    def save_to_pdf(self, md_filepath):
        try:
            import markdown2
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            
            # Read markdown content
            with open(md_filepath, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert to HTML
            html_content = markdown2.markdown(md_content)
            
            # Create full HTML document (for styling)
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>ML Interview Questions</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
                    h2 {{ color: #34495e; margin-top: 30px; }}
                    code {{ background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
                    pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Generate PDF filename
            pdf_filename = md_filepath.replace('.md', '.pdf')
            
            # Convert to PDF
            font_config = FontConfiguration()
            HTML(string=full_html).write_pdf(pdf_filename, font_config=font_config)
            
            print(f"PDF saved to: {pdf_filename}")
            return pdf_filename
            
        except ImportError:
            print("PDF generation requires markdown2 and weasyprint packages.")
            print("Install with: pip install markdown2 weasyprint")
            return None
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return None 