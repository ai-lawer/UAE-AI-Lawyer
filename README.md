# UAE-AI-Lawyer

In this project, we are leveraging Retrieval-Augmented Generation (RAG) with a [vector database](https://drive.google.com/file/d/1L-ejvE8oTeuREYTKLbcDnNX9w0bMOgYZ/view?usp=drive_link) created by OpenAI using [obadabaq/uae-laws](https://huggingface.co/datasets/obadabaq/uae-laws) dataset and a fine-tuned Llama3 [model (obadabaq/ai_lawyer)](https://huggingface.co/obadabaq/ai_lawyer) for legal texts using the [obadabaq/structured-uae-laws](https://huggingface.co/datasets/obadabaq/structured-uae-laws), enabling efficient legal research and accurate advice tailored to UAE laws.

## Repository Contents

* `finetune_unsloth.ipynb`: Jupyter notebook for fine-tuning the model.
* `main.ipynb`: Jupyter notebook using the fine-tuned model & RAG with a vector database.

## Fine-Tuning Details

### Model

* Base Model: [StevenChen16/llama3-8b-Lawyer](https://huggingface.co/StevenChen16/llama3-8b-Lawyer)
* Fine-Tuned Model: [obadabaq/ai_lawyer](https://huggingface.co/obadabaq/ai_lawyer)

### Dataset

* Dataset used: [obadabaq/structured-uae-laws](https://huggingface.co/datasets/obadabaq/structured-uae-laws)

## Usage

Follow the instructions in the main.ipynb file.

The model can engage in a detailed interaction, simulating the behavior of a lawyer. Provide the case, and the model will get the most four relevent laws from the [obadabaq/uae-laws](https://huggingface.co/datasets/obadabaq/uae-laws) to consider them as input to make informed judgments.
The model is traind on these topics:
* economy and business
* family and community
* finance and banking
* industry and technical standardisation
* justice and juiciary, labour
* residency and leberal professions
* security and safety
* tax


### Example Interaction

Enter the question (case)

```
question = input("Enter your legal question\n")
# for example: what are the rights and obligations of an employer and an employee in the event of termination of an employment contract?
```

Query the vector DB

```
content = query_db(question)
```

Generate a Response using the Model and the Vector DB result

```
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    prompt.format(
        question,
        content,
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
```

Output

```
Based on the Federal Decree by Law No. (33) of 2021 Concerning Regulating Labour Relations, Article (42), an employer can terminate an employment contract for various reasons, including the death of the employee or employee's permanent disability, bankruptcy of the employer, or employee's failure to renew their work permit. The employer must provide a ticket for the employee's return to their country and pay any dues owed by the employer. In the event of direct recruitment, the employee bears the cost of returning to their country, but if they are unable to bear this cost, the employer must bear it. If the employee is recruited
```




## Results

Fine-tuning a specialized legal LLM with structured data and RAG with a contextual prompts, has shown promising results in understanding and answering legal questions. By leveraging advanced techniques such as LoRA, UnslothAI optimizations, and OpenAIEmbeddings, the training process was efficient and effective, ensuring a high-quality model output.

## Acknowledgements

* [unsloth](https://github.com/unslothai/unsloth)
* [UAE Legislation](https://uaelegislation.gov.ae/en)
