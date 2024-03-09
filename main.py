import llm
# import google.generativeai as genai
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline
import torch
import os
from datasets import load_dataset


# https://huggingface.co/docs/transformers/main/en/installation
# https://huggingface.co/docs/transformers/en/llm_tutorial
save_path = "./models/t5-small"
save_path_GPT2 = "./models/gpt2"
tokenizer = None

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    # inputs.add_special_tokens({'pad_token': '[PAD]'})

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def save_T5():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    return save_path

def load_T5():
    tokenizer = T5Tokenizer.from_pretrained(save_path)
    model = T5ForConditionalGeneration.from_pretrained(save_path)
    return model, tokenizer


def save_GPT2():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer.save_pretrained(save_path_GPT2)
    model.save_pretrained(save_path_GPT2)
    return save_path_GPT2

def load_GPT2():
    tokenizer = GPT2Tokenizer.from_pretrained(save_path_GPT2)
    model = GPT2LMHeadModel.from_pretrained(save_path_GPT2)
    return model, tokenizer
    
def train(model, tokenizer, name):
    squad = load_dataset("squad", split="train[:5000]")
    squad = squad.train_test_split(test_size=0.2)
    tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

    training_args = TrainingArguments(
        output_dir=f"./model/{name}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

def pipeline_inference(question, context, name):
    question_answerer = pipeline("question-answering", model=name)
    return question_answerer(question=question, context=context)


def query(model, tokenizer, query):
    input_id = tokenizer([query], return_tensors="pt").input_ids
    output = model.generate(input_id, do_sample=True, max_length=100)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

# save_path = save_T5()
# model, tokenizer = load_T5()
# output = query(model, tokenizer, "Hi I am Tom")
# print(output)
# save_GPT2()
# model, tokenizer = load_GPT2()
# output = query(model, tokenizer, "Hi I am Tom")
# print(output)

save_GPT2()
model, tokenizer = load_GPT2()
name = "gpt2test"
train(model, tokenizer, name)
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
output = pipeline_inference(question, context, name)
print(output)


