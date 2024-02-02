import argparse
import os
import dspy
import openai
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# fill trainset with a few examples
my_rag_trainset = [
    dspy.Example(
        question="Who was beginning to get very tired while sitting by her sister?",
        answer="Alice"
    ).with_inputs('question'),
    dspy.Example(
        question="What did Alice find remarkable about the White Rabbit?",
        answer="It took a watch out of its waistcoat-pocket"
    ).with_inputs('question'),
    dspy.Example(
        question="What was Alice's reaction to the Rabbit saying 'Oh dear! Oh dear! I shall be late!'?",
        answer="She thought it seemed quite natural at the time"
    ).with_inputs('question'),
    dspy.Example(
        question="What did Alice decide to do after seeing the Rabbit take out a watch?",
        answer="She ran across the field after it"
    ).with_inputs('question'),
    dspy.Example(
        question="What did Alice find inside the well as she fell?",
        answer="Cupboards and book-shelves with maps and pictures"
    ).with_inputs('question'),
    dspy.Example(
        question="What was labelled 'ORANGE MARMALADE' that Alice found?",
        answer="A jar"
    ).with_inputs('question'),
    dspy.Example(
        question="Why did Alice not want to drop the jar of ORANGE MARMALADE?",
        answer="For fear of killing somebody underneath"
    ).with_inputs('question')
]


# DSPy Configuration
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=turbo)

# ChromaDB Configuration
CHROMA_PATH = "chroma"
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


# defining own RAG program
class RAG(dspy.Module):
    # declares modules you will use
    def __init__(self, num_passages=3):
        super().__init__()
        self.num_passages = num_passages
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
        # DSPy offers general-purpose modules that take the shape of your own sub-tasks — and not pre-built functions for specific applications.
    
    # forward method expresses any computation you want to do with the modules
    def forward(self, question):
        # Retrieve context from ChromaDB
        results = db.similarity_search_with_relevance_scores(question, k=self.num_passages)
        if len(results) == 0 or results[0][1] < 0.7:
            return "Unable to find matching results."
        
        context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            # Ensure context and answer are included in the return value
        answer = self.generate_answer(context=context, question=question).answer  # Get the answer
        return dspy.Prediction(context=context, answer=answer)  # Return both context and answer
       

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    
    # Instantiate RAG and compile it
    rag = RAG()
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
    compiled_rag = teleprompter.compile(rag, trainset=my_rag_trainset)

    # Get the prediction.
    answer = compiled_rag(args.query_text)

    # Print the answer.
    print(f"Question: {args.query_text}")
    print(f"Predicted Answer: {answer}")

# Validation logic (you may need to adjust this part to fit your specific validation needs)
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

if __name__ == "__main__":
    main()
