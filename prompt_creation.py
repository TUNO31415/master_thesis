import pandas as pd
from datasets import Dataset

def create_prompt(speaker_name, partner_name, sentence, history=None):
    base_prompt = """Here, you are asked to rate the interaction you just took part in. We are interested in your personal (subjective) impression of the situation. Thus, we ask you to be as honest as possible and describe the situation by using the following scale: \n 
Strongly disagree = 1, Somewhat disagree = 2, Neither agree nor disagree = 3, Somewhat agree = 4, Strongly Agree = 5

1. What each of us does in this situation affects the other. 
2. Our preferred outcomes in this situation are conflicting. 
3. How we behave now will have consequences for future outcomes.
4. We both know what the other wants.
5. Whatever each of us does in this situation, our actions will not affect the other's outcome.
6. We can both obtain our preferred outcomes.
7. Our future interactions are not affected by the outcomes of this situation.
8. I don't think the other knows what I want.

For each item, please think of the same conversation and indicate how the following statements describe the specific situation.
Definitely person X = 1, Maybe person X = 2, Neither person X nor myself = 3, Maybe myself = 4, Definitely myself = 5

9. Who do you feel had more power to determine their own outcomes in this situation?
10. Who has the least amount of influence on the outcomes of this situation?

Use the template to answer in JSON format. You do not need to provide explanation.
{"Q1" : SCORE, "Q2" : SCORE, "Q3" : SCORE, ... , "Q10" : SCORE}
[/INST]
"""
    if history:
        header = f"Act as Person {speaker_name}. You are now having a conversation with Person {partner_name}. You are asked to answer the following 10 questions at the moment you said \"{sentence}\" given this conversation history. \n{history}\nFrom now on, person X means your conversation partner, Person {partner_name}. <</SYS>>\n"
    else:
        header = f"[INST]<<SYS>> \n Act as Person {speaker_name}. You are now having a conversation with Person {partner_name}. You are asked to answer the following 10 questions at the moment you said \"{sentence}\" at the beginning of the conversation. From now on, person X means your conversation partner, Person {partner_name}. <</SYS>>\n"
    
    return header + base_prompt

def llm_input_generator(df, speaker00_name, speaker01_name):
    def generate_prompts(speaker_df, speaker_name, partner_name):
        prompts = []
        for index, row in speaker_df.iterrows():
            sentence = row["utterance"]
            history = row["dialogue history"] if index != 0 else None
            prompt = create_prompt(speaker_name, partner_name, sentence, history)
            prompts.append(prompt)
        return prompts
    
    speaker_00 = df[df["speaker"] == speaker00_name]
    speaker_01 = df[df["speaker"] == speaker01_name]

    prompts00 = generate_prompts(speaker_00, speaker00_name, speaker01_name)
    prompts01 = generate_prompts(speaker_01, speaker01_name, speaker00_name)

    ds00 = Dataset.from_pandas(pd.DataFrame({"prompt": prompts00}))
    ds01 = Dataset.from_pandas(pd.DataFrame({"prompt": prompts01}))
    
    return ds00, ds01

def create_prompt_without_number(speaker_name, partner_name, sentence, history=None):
    base_prompt = """Here, you are asked to rate the interaction you just took part in. We are interested in your personal (subjective) impression of the situation. Thus, we ask you to be as honest as possible and describe the situation by using the following scale: \n 
Strongly disagree, Somewhat disagree, Neither agree nor disagree, Somewhat agree = 4, Strongly Agree

1. What each of us does in this situation affects the other. 
2. Our preferred outcomes in this situation are conflicting. 
3. How we behave now will have consequences for future outcomes.
4. We both know what the other wants.
5. Whatever each of us does in this situation, our actions will not affect the other's outcome.
6. We can both obtain our preferred outcomes.
7. Our future interactions are not affected by the outcomes of this situation.
8. I don't think the other knows what I want.

For each item, please think of the same conversation and indicate how the following statements describe the specific situation.
Definitely person X, Maybe person X, Neither person X nor myself, Maybe myself, Definitely myself

9. Who do you feel had more power to determine their own outcomes in this situation?
10. Who has the least amount of influence on the outcomes of this situation?

Use the template to answer in JSON format. You do not need to provide explanation.
{"Q1" : SCALE, "Q2" : SCALE, "Q3" : SCALE, ... , "Q10" : SCALE}
[/INST]
"""
    if history:
        header = f"[INST]<<SYS>> \n Act as Person {speaker_name}. You are now having a conversation with Person {partner_name}. You are asked to answer the following 10 questions at the moment you said \"{sentence}\" given this conversation history. \n{history}\nFrom now on, person X means your conversation partner, Person {partner_name}. <</SYS>>\n"
    else:
        header = f"[INST]<<SYS>> \n Act as Person {speaker_name}. You are now having a conversation with Person {partner_name}. You are asked to answer the following 10 questions at the moment you said \"{sentence}\" at the beginning of the conversation. From now on, person X means your conversation partner, Person {partner_name}. <</SYS>>\n"
    
    return header + base_prompt

def llm_input_generator_without_number(df, speaker00_name, speaker01_name):
    def generate_prompts(speaker_df, speaker_name, partner_name):
        prompts = []
        for index, row in speaker_df.iterrows():
            sentence = row["utterance"]
            history = row["dialogue history"] if index != 0 else None
            prompt = create_prompt_without_number(speaker_name, partner_name, sentence, history)
            prompts.append(prompt)
        return prompts
    
    speaker_00 = df[df["speaker"] == speaker00_name]
    speaker_01 = df[df["speaker"] == speaker01_name]

    prompts00 = generate_prompts(speaker_00, speaker00_name, speaker01_name)
    prompts01 = generate_prompts(speaker_01, speaker01_name, speaker00_name)

    ds00 = Dataset.from_pandas(pd.DataFrame({"prompt": prompts00}))
    ds01 = Dataset.from_pandas(pd.DataFrame({"prompt": prompts01}))
    
    return ds00, ds01


def create_prompt_with_context(speaker_name, partner_name, sentence, task_type, history=None):
    base_prompt = """Here, you are asked to rate the interaction you just took part in. We are interested in your personal (subjective) impression of the situation. Thus, we ask you to be as honest as possible and describe the situation by using the following scale: \n 
Strongly disagree, Somewhat disagree, Neither agree nor disagree, Somewhat agree = 4, Strongly Agree

1. What each of us does in this situation affects the other. 
2. Our preferred outcomes in this situation are conflicting. 
3. How we behave now will have consequences for future outcomes.
4. We both know what the other wants.
5. Whatever each of us does in this situation, our actions will not affect the other's outcome.
6. We can both obtain our preferred outcomes.
7. Our future interactions are not affected by the outcomes of this situation.
8. I don't think the other knows what I want.

For each item, please think of the same conversation and indicate how the following statements describe the specific situation.
Definitely person X, Maybe person X, Neither person X nor myself, Maybe myself, Definitely myself

9. Who do you feel had more power to determine their own outcomes in this situation?
10. Who has the least amount of influence on the outcomes of this situation?

Use the template to answer in JSON format. You do not need to provide explanation.
{"Q1" : SCALE, "Q2" : SCALE, "Q3" : SCALE, ... , "Q10" : SCALE}
[/INST]
""" 

    if task_type == "warm":
        task_definition = "a partner to carry out a task with in the future where your partner's warmth is important"
    else:
        task_definition = "a partner to carry out a task with in the future where your partner's competence is important"

    if history:
        header = f"[INST]<<SYS>> \n Act as Person {speaker_name}. You are now having a conversation with Person {partner_name} to find {task_definition}. You are asked to answer the following 10 questions at the moment you said \"{sentence}\" given this conversation history. \n{history}\nFrom now on, person X means your conversation partner, Person {partner_name}. <</SYS>>\n"
    else:
        header = f"[INST]<<SYS>> \n Act as Person {speaker_name}. You are now having a conversation with Person {partner_name} to find {task_definition}. You are asked to answer the following 10 questions at the moment you said \"{sentence}\" at the beginning of the conversation. From now on, person X means your conversation partner, Person {partner_name}. <</SYS>>\n"
    
    return header + base_prompt

def llm_input_generator_with_context(df, speaker00_name, speaker01_name, batch_id):
    def generate_prompts_with_context(speaker_df, speaker_name, partner_name, task_type):
        prompts = []
        for index, row in speaker_df.iterrows():
            sentence = row["utterance"]
            history = row["dialogue history"] if index != 0 else None
            prompt = create_prompt_with_context(speaker_name, partner_name, sentence, task_type, history)
            prompts.append(prompt)
        return prompts
    
    col_df = pd.read_csv("/tudelft.net/staff-umbrella/tunoMSc2023/paco_dataset/Coordination/curated_coord_data_ffinal.csv")
    col_filtered_df = col_df[col_df["batchID"] == batch_id]
    taskType = col_filtered_df["coord_taskType"].tolist()[0]
    print(taskType)

    
    speaker_00 = df[df["speaker"] == speaker00_name]
    speaker_01 = df[df["speaker"] == speaker01_name]

    prompts00 = generate_prompts_with_context(speaker_00, speaker00_name, speaker01_name, taskType)
    prompts01 = generate_prompts_with_context(speaker_01, speaker01_name, speaker00_name, taskType)

    ds00 = Dataset.from_pandas(pd.DataFrame({"prompt": prompts00}))
    ds01 = Dataset.from_pandas(pd.DataFrame({"prompt": prompts01}))
    
    return ds00, ds01

def create_prompt_per_question(speaker_name, partner_name, sentence, history=None):

    question_list = [
        "What each of us does in this situation affects the other.",
        "Our preferred outcomes in this situation are conflicting. ",
        "How we behave now will have consequences for future outcomes.",
        "We both know what the other wants.",
        "Whatever each of us does in this situation, our actions will not affect the other's outcome.",
        "We can both obtain our preferred outcomes.",
        "Our future interactions are not affected by the outcomes of this situation.",
        "I don't think the other knows what I want.",
        "Who do you feel had more power to determine their own outcomes in this situation?",
        "Who has the least amount of influence on the outcomes of this situation?"
    ]

    base_q1_8 = "Here, you are asked to rate the interaction you just took part in. For the following statement, we are interested in your personal (subjective) impression of the situation. Thus, we ask you to be as honest as possible and describe the situation by using the following scale: \n Strongly disagree, Somewhat disagree, Neither agree nor disagree, Somewhat agree, Strongly Agree \n"
    base_q9_10 = "Here, you are asked to rate the interaction you just took part in. Please think of the conversation you just took part in and indicate how the following statement describe the specific situation by using the following scale: \n Definitely person X, Maybe person X, Neither person X nor myself, Maybe myself, Definitely myself \n"

    prompts = []

    for i, question in enumerate(question_list):
        ending = f"Statement : \"{question}\" \n Only answer the scale. You do not need to provide explanations. [/INST]"
        if history:
            header = f"[INST]<<SYS>> \n Act as Person {speaker_name}. You are now having a conversation with Person {partner_name}. You are asked to answer the following 10 questions at the moment you said \"{sentence}\" given this conversation history. \n{history}\nFrom now on, person X means your conversation partner, Person {partner_name}. <</SYS>>\n"
            if i < 8:
                prompts.append(header + base_q1_8 + ending)
            else:
                prompts.append(header + base_q9_10 + ending)
        else:
            header = f"[INST]<<SYS>> \n Act as Person {speaker_name}. You are now having a conversation with Person {partner_name}. You are asked to answer the following 10 questions at the moment you said \"{sentence}\" at the beginning of the conversation. From now on, person X means your conversation partner, Person {partner_name}. <</SYS>>\n"
            if i < 9:
                prompts.append(header + base_q1_8 + ending)
            else:
                prompts.append(header + base_q9_10 + ending)

    return prompts

def llm_input_generator_per_questions(df, speaker00_name, speaker01_name):
    def generate_prompts(speaker_df, speaker_name, partner_name):
        prompts = []
        for index, row in speaker_df.iterrows():
            sentence = row["utterance"]
            history = row["dialogue history"] if index != 0 else None
            prompt_qs = create_prompt_per_question(speaker_name, partner_name, sentence, history)
            prompts += prompt_qs
        return prompts
    
    speaker_00 = df[df["speaker"] == speaker00_name]
    speaker_01 = df[df["speaker"] == speaker01_name]

    prompts00 = generate_prompts(speaker_00, speaker00_name, speaker01_name)
    prompts01 = generate_prompts(speaker_01, speaker01_name, speaker00_name)

    ds00 = Dataset.from_pandas(pd.DataFrame({"prompt": prompts00}))
    ds01 = Dataset.from_pandas(pd.DataFrame({"prompt": prompts01}))
    
    return ds00, ds01

def create_prompt_summary(speaker_name, partner_name, history):
    base_prompt = """Here, you are asked to rate the interaction you just took part in. We are interested in your personal (subjective) impression of the situation. Thus, we ask you to be as honest as possible and describe the situation by using the following scale: \n 
Strongly disagree, Somewhat disagree, Neither agree nor disagree, Somewhat agree = 4, Strongly Agree

1. What each of us does in this situation affects the other. 
2. Our preferred outcomes in this situation are conflicting. 
3. How we behave now will have consequences for future outcomes.
4. We both know what the other wants.
5. Whatever each of us does in this situation, our actions will not affect the other's outcome.
6. We can both obtain our preferred outcomes.
7. Our future interactions are not affected by the outcomes of this situation.
8. I don't think the other knows what I want.

For each item, please think of the same conversation and indicate how the following statements describe the specific situation.
Definitely person X, Maybe person X, Neither person X nor myself, Maybe myself, Definitely myself

9. Who do you feel had more power to determine their own outcomes in this situation?
10. Who has the least amount of influence on the outcomes of this situation?

Use the template to answer in JSON format. You do not need to provide explanation.
{"Q1" : SCALE, "Q2" : SCALE, "Q3" : SCALE, ... , "Q10" : SCALE}
[/INST]
"""
    header = f"[INST]<<SYS>> \n Act as Person {speaker_name}. You are now having a conversation with Person {partner_name}. You are asked to answer the following 10 questions after the following conversation \n {history}. \n From now on, person X means your conversation partner, Person {partner_name}. <</SYS>>\n"
    
    return header + base_prompt

def llm_input_generator_summary(df, speaker00_name, speaker01_name):
    def generate_prompts(speaker_df, speaker_name, partner_name):
        row = speaker_df.iloc[-1]
        history = row["dialogue history"]
        prompt = create_prompt_summary(speaker_name, partner_name, history)
        return [prompt]

    prompts00 = generate_prompts(df, speaker00_name, speaker01_name)
    prompts01 = generate_prompts(df, speaker01_name, speaker00_name)

    ds00 = Dataset.from_pandas(pd.DataFrame({"prompt": prompts00}))
    ds01 = Dataset.from_pandas(pd.DataFrame({"prompt": prompts01}))
    
    return ds00, ds01

