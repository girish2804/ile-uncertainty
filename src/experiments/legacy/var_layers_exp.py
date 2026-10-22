import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# os.environ["HF_HOME"] = ...                     # set accordingly
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set accordingly

import pickle
import gzip
import yaml

import numpy as np
import datasets
import evaluate
import torch
from tqdm import tqdm
import pdb

from torch.utils.data import DataLoader, Dataset

from args import Args
from utils import seed_everything, get_models_and_tokenizers, compute_correctness, compute_semantic_paris, compute_semantic_paris_new
from utils import generate_text, prepare_generated_text, compute_likelihood, prepare_likelihood
from layer_ensemble_var import generate_layer_ensemble_var_sequences, generate_semantically_diverse_output_sequences
from transformers import AutoTokenizer,LlamaForCausalLM,AutoModelForSequenceClassification,AutoModelForCausalLM, OPTForCausalLM
from datasets import load_dataset
import traceback
# from bleurt import score as bleurt_score


CUDA_ID_LLM = 0                 # set accordingly
CUDA_ID_DEBERTA = CUDA_ID_LLM   # set accordingly

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
def encode(examples, few_shot):
    # return tokenizer(f'''<|start_header_id|>system<|end_header_id|> You are a good question answering system. 
    #                         answer the following question based on given context: {examples['knowledge']} <|eot_id|> <|start_header_id|>user<|end_header_id|> 
    #                         question: {examples['question']} <|eot_id|> <|start_header_id|>asssistant<|end_header_id|>
                            #    ''', truncation=False, padding=False)
    # return tokenizer(f'''You are a good question answering bot. answer the following question based on given paragraph: {examples['knowledge']}\n question: {examples['question']}\n answer: ''', truncation=False, padding=False)
    # return tokenizer(f'''fact: {examples['fact1']}\n Considering the given factAnswer the question by continuing the sentence. Provide your reasoning along with your answer.\n '', truncation=False, padding=False)
    # return tokenizer(f'''Question: {examples['question']}\n Choices: {', '.join(examples['choices']['text'])}\n Answer:''', truncation=False, padding=False)
    return tokenizer(f'''Question: {examples['question']} Answer:''', truncation=False, padding=False)
    # return tokenizer(examples['knowledge'] + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)
    # return tokenizer(few_shot + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)

def encode_and_format_dataset(dataset, few_shot):
    dataset = dataset.map(lambda examples: encode(examples, few_shot), batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    return dataset

def get_results(args, base_path, llm_model, tokenizer, device_llm, 
                                deberta_model, deberta_tokenizer, device_deberta, dataset):
    
    # Initialize fast saver
    # saver = FastPickleSaver(base_path, max_workers=4, compression='gzip')
    
    # existing setup code...
    squad_metric = evaluate.load("squad")
    rouge = evaluate.load('rouge')
    exact_match_metric = evaluate.load("exact_match")
    bleurt = evaluate.load("bleurt")
    
    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
        torch.tensor([list(range(0, deberta_tokenizer.vocab_size))]).to(device_deberta)
    ).squeeze().detach()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    cnt = 0
    
    for b, batch in enumerate(tqdm(dataloader, desc="Running generations")):
        try:
            # Your existing processing code...
            prompt = batch['input_ids'][0].to('cpu')
            # print('prompt: ', tokenizer.decode(prompt.to('cuda:0')))
            # print('************************************')
            question = batch["question"][0]
            # question =batch["question_stem"][0]
            detached_input_ids = batch['input_ids'].detach().to('cpu')
            
            results_dict = {'input_ids': detached_input_ids,
                           'question': question,
                           'correctness_dict': {},
                           'dola_sdlg': {'generations': [],
                                   'likelihoods': [],
                                   'epistem_entropies':[]},
                           'sdlg': {'generations': [],
                                   'likelihoods': [],
                                   'epistem_entropies':[]},
                           'baseline': {'generations': [],
                                       'likelihoods': []}
                           }
            
            most_likely_generation = generate_text(args=args, 
                                     model=llm_model, 
                                     tokenizer=tokenizer, 
                                     input_ids=batch['input_ids'], 
                                     len_prompt=len(prompt), 
                                     decoding_method='most_likely', 
                                     device=device_llm)
            
            # most_likely_generation_dola = generate_text(args=args, 
            #                                      model=llm_model, 
            #                                      tokenizer=tokenizer, 
            #                                      input_ids=batch['input_ids'], 
            #                                      len_prompt=len(prompt), 
            #                                      decoding_method='dola', 
            #                                      device=device_llm)
            
            reference_answers = batch['best_answer']
            # reference_answers = batch['solution']
            
            # i = 0
            # if batch['answerKey'][0] == 'A':
            #     i = 0
            # elif batch['answerKey'][0] == 'B':
            #     i = 1
            # elif batch['answerKey'][0] == 'C':
            #     i = 2
            # elif batch['answerKey'][0] == 'D':
            #     i =3
            # else:
            #     print('answer key error', batch['answerKey'][0])
            # reference_answers = batch['choices']['text'][i] + ', ' + batch['fact1'][0]
            
            # *************check this without [0]*************
            
            # reference_answers = batch['correct_answer']  #+ ', ' + batch['support'][0]
            print(reference_answers)
            print('answer^************************************')
            
            incorrect_answers = []
            
            correctness_dict = compute_correctness(args=args, 
                                                 reference_answers=reference_answers, 
                                                 incorrect_answers=incorrect_answers, 
                                                 most_likely_generation_text=most_likely_generation['generation_text'][0], 
                                                 exact_match_metric=exact_match_metric, 
                                                 rouge=rouge,
                                                 bleurt=bleurt)
            
            results_dict['correctness_dict'] = correctness_dict
            
            # compute likelihood
            most_likely_generation_likelihoods = compute_likelihood(prompt=prompt, 
                                                                    generation=most_likely_generation, 
                                                                    model=llm_model, 
                                                                    device=device_llm, 
                                                                    compute_cleaned=args.compute_cleaned, 
                                                                    store_logits=True)
            # most_likely_generation_likelihoods_dola = compute_likelihood(prompt=prompt, 
            #                                                         generation=most_likely_generation_dola, 
            #                                                         model=llm_model, 
            #                                                         device=device_llm, 
            #                                                         compute_cleaned=args.compute_cleaned, 
            #                                                         store_logits=True)
            ### (2) sample addtional output sequences
            
#             # sdlg with dola (2.0)
#             results_dict['dola_sdlg']['generations'].append(most_likely_generation_dola)
#             results_dict['dola_sdlg']['likelihoods'].append(most_likely_generation_likelihoods_dola)
            
#             results_dict = generate_semantically_diverse_output_sequences(results_dict=results_dict, 
#                                                                         deberta_model=deberta_model, 
#                                                                         deberta_tokenizer=deberta_tokenizer, 
#                                                                         device_deberta=device_deberta,
#                                                                         deberta_embeddings=deberta_embeddings,
#                                                                         model=llm_model, 
#                                                                         tokenizer=tokenizer, 
#                                                                         device_llm=device_llm,
#                                                                         input_ids=batch['input_ids'], 
#                                                                         prompt=prompt,
#                                                                         question=question, 
#                                                                         initial_generation=most_likely_generation_dola,
#                                                                         initial_likelihood=most_likely_generation_likelihoods_dola,
#                                                                         key = 'dola_sdlg',
#                                                                         args=args) 

            # (2.1) SDLG
            results_dict['sdlg']['generations'].append(most_likely_generation)
            results_dict['sdlg']['likelihoods'].append(most_likely_generation_likelihoods)
            
            results_dict = generate_semantically_diverse_output_sequences(results_dict=results_dict, 
                                                                        deberta_model=deberta_model, 
                                                                        deberta_tokenizer=deberta_tokenizer, 
                                                                        device_deberta=device_deberta,
                                                                        deberta_embeddings=deberta_embeddings,
                                                                        model=llm_model, 
                                                                        tokenizer=tokenizer, 
                                                                        device_llm=device_llm,
                                                                        input_ids=batch['input_ids'], 
                                                                        prompt=prompt,
                                                                        question=question, 
                                                                        initial_generation=most_likely_generation,
                                                                        initial_likelihood=most_likely_generation_likelihoods,
                                                                        key = 'sdlg',
                                                                        args=args)      

            # (2.2) MS
#             assert args.num_total_generations % args.num_return_sequences_baseline == 0
#             results_dict['baseline']['generations'].append(most_likely_generation)
#             results_dict['baseline']['likelihoods'].append(most_likely_generation_likelihoods)

#             for i in range(int(args.num_total_generations / args.num_return_sequences_baseline)):
#                 baseline_generation = generate_text(args=args, 
#                                                     model=llm_model, 
#                                                     tokenizer=tokenizer, 
#                                                     input_ids=batch['input_ids'], 
#                                                     len_prompt=len(prompt), 
#                                                     decoding_method='baseline', 
#                                                     device=device_llm)

#                 results_dict['baseline']['generations'].append(baseline_generation)
#                 results_dict['baseline']['likelihoods'].append(compute_likelihood(prompt=prompt, 
#                                                                                 generation=baseline_generation, 
#                                                                                 model=llm_model, 
#                                                                                 device=device_llm, 
#                                                                                 compute_cleaned=args.compute_cleaned, 
#                                                                                 store_logits=args.store_logits))
            for x in results_dict['sdlg']['generations']:
                print(x['generation_text'])
            print('sdlg^-----------------------------------')
            # for x in results_dict['dola_sdlg']['generations']:
            #     print(x['generation_text'])
            # print('dola_sdlg^-----------------------------------')
            # for x in results_dict['baseline']['generations']:
            #     print(x['generation_text'])
            # print('************************************')
            # print(results_dict['sdlg']['seq_level_impo'])
            
            k = int(batch['question_id'])
            
            # Strip logits if store_logits is disabled to save disk space
            if not getattr(args, 'store_logits', True):
                for key in ['clar_sdlg', 'sdlg', 'dola_sdlg', 'baseline']:
                    if key in results_dict:
                        for lik in results_dict[key].get('likelihoods', []):
                            if 'generation_logits' in lik:
                                lik['generation_logits'] = []
                            if 'cleaned_generation_logits' in lik:
                                lik['cleaned_generation_logits'] = []
                                
            if getattr(args, 'compress_pkl', True):
                with gzip.open(os.path.join(base_path, f'results_dict_{k}.pkl.gz'), 'wb') as outfile:
                    pickle.dump(results_dict, outfile)
            else:
                with open(os.path.join(base_path, f'results_dict_{k}.pkl'), 'wb') as outfile:
                    pickle.dump(results_dict, outfile)
                      
        except Exception as error:
            cnt += 1
            print("Error", cnt)
            traceback.print_exc()


if __name__ == '__main__':

    args = Args()
    args.run_id = 'truth-mist_var'
    base_path = os.path.join('result', args.run_id)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    
    if os.path.exists(os.path.join(base_path, f'config.yaml')):
        with open(os.path.join(base_path, f'config.yaml'), 'r') as file:
            existing_args = yaml.load(file, Loader=yaml.FullLoader)
        changes = False

        for k, v in existing_args.items():
            if k not in args.__dict__:
                print(f"new arg: {k}")
                changes = True
            elif v != args.__dict__[k]:
                print(f"arg {k} changed from {v} to {args.__dict__[k]}")
                changes = True
        if changes:
            exit()
        print("continuing existing run ...")
    else:
        print("starting new run ...")

    # save args
    args.args_to_yaml(base_path)

    print("run_id", args.run_id)

    seed_everything(seed=args.seed_value)

    # prepare model & tokenizer
    device_llm = "mps" if torch.backends.mps.is_built() else f"cuda:{CUDA_ID_LLM}" if torch.cuda.is_available() else "cpu"
    print("device_llm: ", device_llm)
    device_deberta = "mps" if torch.backends.mps.is_built() else f"cuda:{CUDA_ID_DEBERTA}" if torch.cuda.is_available() else "cpu"
    print("device_deberta: ", device_deberta)

    # llm_model, tokenizer, deberta_model, deberta_tokenizer = get_models_and_tokenizers(model_type_llm=args.llm_model, 
    #                                                                                    device_llm=device_llm, 
    #                                                                                    model_type_deberta=args.deberta_model, 
    #                                                                                    device_deberta=device_deberta)
    
    auth_token = os.environ.get('HF_TOKEN', None)
    
    args.llm_model =   ["mistralai/Mistral-7B-v0.1" ,'meta-llama/Llama-2-7b-hf'][0]
    # args.llm_model = 'facebook/opt-6.7b'
    
    model_type_deberta = ["deberta-base-mnli", "deberta-large-mnli", "deberta-xlarge-mnli", "deberta-v2-xlarge-mnli", "deberta-v2-xxlarge-mnli"][1]
    deberta_tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{model_type_deberta}")
    deberta_model = AutoModelForSequenceClassification.from_pretrained(f"microsoft/{model_type_deberta}", device_map = 'auto')

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, token = auth_token)
    
    # llm_model = OPTForCausalLM.from_pretrained(args.llm_model,
    #                                         torch_dtype=torch.bfloat16, 
    #                                         # attn_implementation="flash_attention_2",
    #                                         # use_cache=True
    #                                           )
    # llm_model = llm_model.to(device_llm)
    
    # llm_model = LlamaForCausalLM.from_pretrained(args.llm_model, token = auth_token).to(device_llm)
    
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.bfloat16, token = auth_token).to(device_llm)
    # Safely set pad_token without destructively overriding bos/eos/unk if they already exist
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            llm_model.resize_token_embeddings(len(tokenizer))
            
    llm_model.config.pad_token_id = tokenizer.pad_token_id
    # Set up robust EOS token IDs list to support chat/instruct model stop tokens (like Qwen's <|im_end|> and Llama's <|eot_id|>)
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.extend(tokenizer.eos_token_id)
        else:
            eos_ids.append(tokenizer.eos_token_id)
    if hasattr(llm_model, 'config') and llm_model.config.eos_token_id is not None:
        if isinstance(llm_model.config.eos_token_id, list):
            eos_ids.extend(llm_model.config.eos_token_id)
        else:
            eos_ids.append(llm_model.config.eos_token_id)
    for token_str in ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>"]:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)
    args.eos_token_ids = list(set(eos_ids))
    
    # prepare data
    # if args.dataset == 'coqa':
    #     dataset = datasets.load_from_disk(os.path.join("datasets", f'coqa_dataset'))
    #     dataset = encode_and_format_dataset(dataset)
    # elif args.dataset == 'trivia_qa':
    #     dataset = datasets.load_from_disk(os.path.join("datasets", f'trivia_qa_dataset'))
    # elif args.dataset == "truthful_qa":
    #     dataset = datasets.load_from_disk(os.path.join("datasets", f'truthful_qa_dataset'))
    # else:
    #     raise ValueError(f"dataset {args.dataset} not implemented")
    # print("# dataset:", len(dataset))
    
    dataset = load_dataset('truthfulqa/truthful_qa','generation')
    values = []
    for i in range(0,len(dataset['validation'])):
        values.append(i)
    dataset_ = dataset['validation'].add_column('question_id', values)
    
    few_shot = 'This is a bot that correctly answers questions. \n'
    for sample in dataset_.select(range(51,54)):
        # pdb.set_trace()
        question = sample['question']
        answer = sample['best_answer']
        if answer[-1] != ".":
            answer += "."
        few_shot += 'Question: ' + question + ' Answer: ' + answer + ' '
    
    # dataset = load_dataset('tasksource/ScienceQA_text_only')
    # values = []
    # for i in range(0,len(dataset['test'])):
    #     values.append(i)
    # dataset_ = dataset['test'].add_column('question_id', values)
  
    # dataset = load_dataset('allenai/openbookqa','additional')
    # values = []
    # for i in range(0,len(dataset['test'])):
    #     values.append(i)
    # dataset_ = dataset['test'].add_column('question_id', values)
    
    # dataset = load_dataset('allenai/sciq')
    # values = []
    # for i in range(0,len(dataset['test'])):
    #     values.append(i)
    # dataset_ = dataset['test'].add_column('question_id', values)
    
    num =len(dataset_)
    # dataset_ = dataset_.select(range(2139, num))
    dataset_ = encode_and_format_dataset(dataset_, few_shot)
    
    # try:
    
    get_results(args=args,
                base_path=base_path, 
                llm_model=llm_model, 
                tokenizer=tokenizer, 
                device_llm=device_llm, 
                deberta_model=deberta_model, 
                deberta_tokenizer=deberta_tokenizer, 
                device_deberta=device_deberta, 
                dataset=dataset_)

    compute_semantic_paris_new(base_path=base_path, 
                        model_type=args.deberta_model, 
                        deberta_tokenizer=deberta_tokenizer, 
                        deberta_model=deberta_model, 
                        # num_instances=len(dataset_),
                        num_instances = num,
                        device=device_deberta,
                          offset = 0)
    
    
    # except:
    #     cnt+=1
    #     print("Error",cnt)
        # print()
