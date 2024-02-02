import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import dspy
import os
import openai
from dspy.teleprompt import BootstrapFewShot
from dspy.datasets import HotPotQA
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")

turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)

# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


# defining a simple RAG program
class RAG(dspy.Module):
    '''
    __init__ method declares modules you will use. RAG will use the built-in Retrieve for retrival 
    and ChainOfThought for generating answers. 
    DSPy offers general pupose mopdules that take shape of your own subtasks,
    not pre-built functions for specific applications.
    
    Modules that use the LM, like ChainOfThought, require a signature. 
    That is a declarative spec that tells the module what it's expected to do. 
    In this example, we use the short-hand signature notation context, 
    question -> answer to tell ChainOfThought it will be given some context and 
    a question and must produce an answer.
    '''
    
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        
    
    '''
    The forward method expresses any computation you want to do with your modules. 
    In this case, we use the module self.retrieve to search for some context and then 
    use the module self.generate_answer, which uses the context and question to 
    generate the answer.
    '''
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)


def main():
    compiled_rag = teleprompter.compile(RAG(), trainset=trainset)
    # Ask any question you like to this simple RAG program.
    my_question = "What castle did David Gregory inherit?"

    # Get the prediction. This contains `pred.context` and `pred.answer`.
    pred = compiled_rag(my_question)

    # Print the contexts and the answer.
    print(f"Question: {my_question}")
    print(f"Predicted Answer: {pred.answer}")
    print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")
    
    
    
    

if __name__ == "__main__":
    main()
