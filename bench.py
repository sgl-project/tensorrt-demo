import time
import json
import argparse

import torch

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def extract_assistant_response(output_text):
    """Model-specific code to extract model responses.

    See this doc for LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/."""
    # Split the output text by the assistant header token
    parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>")

    if len(parts) > 1:
        # Join the parts after the first occurrence of the assistant header token
        response = parts[1].split("<|eot_id|>")[0].strip()

        # Remove any remaining special tokens and whitespace
        response = response.replace("<|eot_id|>", "").strip()

        return response
    else:
        return output_text


QUESTIONS = [
    # Generic assistant questions
    "What are you?",
    "What can you do?",
    # Coding
    "Implement a Python function to compute the Fibonacci numbers.",
    "Write a Rust function that performs binary exponentiation.",
    "How do I allocate memory in C?",
    "What are the differences between Javascript and Python?",
    "How do I find invalid indices in Postgres?",
    "How can you implement a LRU (Least Recently Used) cache in Python?",
    "What approach would you use to detect and prevent race conditions in a multithreaded application?",
    "Can you explain how a decision tree algorithm works in machine learning?",
    "How would you design a simple key-value store database from scratch?",
    "How do you handle deadlock situations in concurrent programming?",
    "What is the logic behind the A* search algorithm, and where is it used?",
    "How can you design an efficient autocomplete system?",
    "What approach would you take to design a secure session management system in a web application?",
    "How would you handle collision in a hash table?",
    "How can you implement a load balancer for a distributed system?",
    "Implement a Python class for a doubly linked list.",
    "Write a Haskell function that generates prime numbers using the Sieve of Eratosthenes.",
    "Develop a simple HTTP server in Rust.",
    # Literate and creative writing
    "What is the fable involving a fox and grapes?",
    "Who does Harry turn into a balloon?",
    "Write a story in the style of James Joyce about a trip to the Australian outback in 2083 to see robots in the beautiful desert.",
    "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
    "Describe a day in the life of a secret agent who's also a full-time parent.",
    "Create a story about a detective who can communicate with animals.",
    "What is the most unusual thing about living in a city floating in the clouds?",
    "In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
    "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
    "Tell a story about a musician who discovers that their music has magical powers.",
    "In a world where people age backwards, describe the life of a 5-year-old man.",
    "Create a tale about a painter whose artwork comes to life every night.",
    "What happens when a poet's verses start to predict future events?",
    "Imagine a world where books can talk. How does a librarian handle them?",
    "Tell a story about an astronaut who discovered a planet populated by plants.",
    "Describe the journey of a letter traveling through the most sophisticated postal service ever.",
    "Write a tale about a chef whose food can evoke memories from the eater's past.",
    "Write a poem in the style of Walt Whitman about the modern digital world.",
    "Create a short story about a society where people can only speak in metaphors.",
    "What are the main themes in Dostoevsky's 'Crime and Punishment'?",
    # History and Philosophy
    "What were the major contributing factors to the fall of the Roman Empire?",
    "How did the invention of the printing press revolutionize European society?",
    "What are the effects of quantitative easing?",
    "How did the Greek philosophers influence economic thought in the ancient world?",
    "What were the economic and philosophical factors that led to the fall of the Soviet Union?",
    "How did decolonization in the 20th century change the geopolitical map?",
    "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
    "What led to the rise and fall of the Mongol Empire?",
    "Discuss the effects of the Industrial Revolution on urban development in 19th century Europe.",
    "How did the Treaty of Versailles contribute to the outbreak of World War II?",
    "What led to the rise and fall of the Mongol Empire?",
    "Discuss the effects of the Industrial Revolution on urban development in 19th century Europe.",
    "How did the Treaty of Versailles contribute to the outbreak of World War II?",
    "Explain the concept of 'tabula rasa' in John Locke's philosophy.",
    "What does Nietzsche mean by 'ressentiment'?",
    "Compare and contrast the early and late works of Ludwig Wittgenstein. Which do you prefer?",
    "How does the trolley problem explore the ethics of decision-making in critical situations?",
    # Thoughtfulness
    "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
    "In a dystopian future where water is the most valuable commodity, how would society function?",
    "If a scientist discovers immortality, how could this impact society, economy, and the environment?",
    "What could be the potential implications of contact with an advanced alien civilization?",
    "Describe how you would mediate a conflict between two roommates about doing the dishes using techniques of non-violent communication.",
    "If you could design a school curriculum for the future, what subjects would you include to prepare students for the next 50 years?",
    "How would society change if teleportation was invented and widely accessible?",
    "Consider a future where artificial intelligence governs countries. What are the potential benefits and pitfalls?",
    # Math
    "What is the product of 9 and 8?",
    "If a train travels 120 kilometers in 2 hours, what is its average speed?",
    "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
    "Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
    "Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
    "Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
    # Facts
    "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
    "What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
    "What was Project A119 and what were its objectives?",
    "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
    "What is the 'Emu War' that took place in Australia in the 1930s?",
    "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
    "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
    "What are 'zombie stars' in the context of astronomy?",
    "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
    "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
    "Which countries in the European Union use currencies other than the Euro, and what are those currencies?",
    # Multilingual
    "战国时期最重要的人物是谁?",
    "Tuende hatua kwa hatua. Hesabu jumla ya mfululizo wa kihesabu wenye neno la kwanza 2, neno la mwisho 42, na jumla ya maneno 21.",
    "Kannst du die wichtigsten Eigenschaften und Funktionen des NMDA-Rezeptors beschreiben?",
    "¿Cuáles son los principales impactos ambientales de la deforestación en la Amazonía?",
    "Décris la structure et le rôle de la mitochondrie dans une cellule.",
    "Какие были социальные последствия Перестройки в Советском Союзе?",
    # Economics and Business
    "What are the principles of behavioral economics and how do they influence consumer choices?",
    "Discuss the impact of blockchain technology on traditional banking systems.",
    "What are the long-term effects of trade wars on global economic stability?",
    "What is the law of supply and demand?",
    "Explain the concept of inflation and its typical causes.",
    "What is a trade deficit, and why does it matter?",
    "How do interest rates affect consumer spending and saving?",
    "What is GDP and why is it important for measuring economic health?",
    "What is the difference between revenue and profit?",
    "Describe the role of a business plan in startup success.",
    "How does market segmentation benefit a company?",
    "Explain the concept of brand equity.",
    "What are the advantages of franchising a business?",
    "What are Michael Porter's five forces and how do they impact strategy for tech startups?",
    # Science and Technology
    "Discuss the potential impacts of quantum computing on data security.",
    "How could CRISPR technology change the future of medical treatments?",
    "Explain the significance of graphene in the development of future electronics.",
    "How do renewable energy sources compare to fossil fuels in terms of environmental impact?",
    "What are the most promising technologies for carbon capture and storage?",
    "Explain why the sky is blue.",
    "What is the principle behind the operation of a microwave oven?",
    "How does Newton's third law apply to rocket propulsion?",
    "What causes iron to rust?",
    "Describe the process of photosynthesis in simple terms.",
    "What is the role of a catalyst in a chemical reaction?",
    "What is the basic structure of a DNA molecule?",
    "How do vaccines work to protect the body from disease?",
    "Explain the significance of mitosis in cellular reproduction.",
    "What are tectonic plates and how do they affect earthquakes?",
    "How does the greenhouse effect contribute to global warming?",
    "Describe the water cycle and its importance to Earth's climate.",
    "What causes the phases of the Moon?",
    "How do black holes form?",
    "Explain the significance of the Big Bang theory.",
    "What is the function of the CPU in a computer system?",
    "Explain the difference between RAM and ROM.",
    "How does a solid-state drive (SSD) differ from a hard disk drive (HDD)?",
    "What role does the motherboard play in a computer system?",
    "Describe the purpose and function of a GPU.",
    "What is TensorRT? What role does it play in neural network inference?",
]


class TRT_Model:

    def __init__(self, model_trt_engine):
        print("Initializing TRT-LLM engine")
        start_time = time.time()

        runner_kwargs = dict(
            engine_dir="{}".format(model_trt_engine),
            lora_dir=None,
            # this will need to be adjusted to use multiple GPUs
            rank=tensorrt_llm.mpi_rank(),
        )

        self.model = ModelRunner.from_dir(**runner_kwargs)

        print("  -- Finished initializing in {} secs".format(time.time() -
                                                             start_time))

    def generate(self, inputs_t, settings):
        return self.model.generate(inputs_t, **settings)


def process_outputs(prompts, tokenizer, outputs_t):
    outputs_text = tokenizer.batch_decode(
        outputs_t[:, 0])  # only one output per input, so we index with 0

    responses = [
        extract_assistant_response(output_text) for output_text in outputs_text
    ]

    # num_tokens = sum(
    #     map(lambda r: len(self.tokenizer.encode(r)), responses))

    for prompt, response in zip(prompts, responses):
        print(
            f"\n\n{prompt}\n\n",
            f"\n\n{response}\n\n",
        )
        time.sleep(0.01)  # to avoid log truncation


def trt_generate(tokenizer, model, settings, chat_prompts, log=False):
    inputs_t = tokenizer(chat_prompts,
                         return_tensors="pt",
                         padding=True,
                         truncation=False)["input_ids"]

    if log:
        print("  -- inputs_t.shape = {}".format(inputs_t.shape))

    outputs_t = model.generate(inputs_t, settings)

    outputs_text = tokenizer.batch_decode(
        outputs_t[:, 0])  # only one output per input, so we index with 0

    responses = [
        extract_assistant_response(output_text) for output_text in outputs_text
    ]

    return responses


def trt_bench(args, model_config, prompts):
    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"])
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Add chat template to prompts
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": prompt
            }],
            add_generation_prompt=True,
            tokenize=False,
        ) for prompt in prompts
    ]

    # Init TRT settings
    trt_settings = dict(
        temperature=args.temperature,
        top_k=args.top_k,
        stop_words_list=None,
        repetition_penalty=args.repetition_penalty,
    )

    trt_settings["max_new_tokens"] = args.output_tokens
    trt_settings["end_id"] = tokenizer.eos_token_id
    trt_settings["pad_id"] = tokenizer.pad_token_id

    # Init TRT Engine
    trt_model = TRT_Model(model_config["model_trt_engine"])

    # Warmup
    for i in range(args.num_warmup):
        outputs = trt_generate(tokenizer, trt_model, trt_settings,
                               chat_prompts, i == 0)

    # Bench
    start_time = time.time()
    for i in range(args.num_bench):
        outputs = trt_generate(tokenizer, trt_model, trt_settings,
                               chat_prompts)
    elapsed_time = time.time() - start_time
    print("  -- TRT generate avg-time: {}".format(elapsed_time /
                                                  args.num_bench))

    for prompt, output in zip(chat_prompts, outputs):
        print(
            f"\n\n{prompt}\n\n",
            f"\n\n{output}\n\n",
        )
        break


def vllm_bench(args, model_config, prompts):
    # Init VLLM settings
    vllm_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        stop_token_ids=None,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.output_tokens)

    # Init VLLM Engine
    vllm_model = LLM(model=model_config["model_id"],
                     dtype=model_config["dtype"],
                     max_seq_len_to_capture=model_config["max_input_len"],
                     tokenizer=model_config["model_id"])

    # Apply chat template
    tokenizer = vllm_model.get_tokenizer()
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": prompt
            }],
            add_generation_prompt=True,
            tokenize=False,
        ) for prompt in prompts
    ]

    # Warmup
    for i in range(args.num_warmup):
        outputs = vllm_model.generate(chat_prompts, vllm_sampling_params)

    # Bench
    start_time = time.time()
    for i in range(args.num_bench):
        outputs = vllm_model.generate(chat_prompts, vllm_sampling_params)
    elapsed_time = time.time() - start_time
    print("  -- VLLM generate avg-time: {}".format(elapsed_time /
                                                   args.num_bench))

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("Prompt [#token_ids = {}]:\n{}\n\nGenerated text:\n{}\n".format(
            len(output.prompt_token_ids), prompt, generated_text))
        break


def main(args):
    print("Reading model_config from: {}".format(args.model_config))
    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    if args.output_tokens > model_config["max_output_len"]:
        raise ValueError("output_tokens={} exceeds max_output_len={}".format(
            args.output_tokens, model_config["max_output_len"]))

    # Init prompts
    question = "Who are you?" * args.prompt_scale
    prompts = []
    for i in range(args.batch_size):
        prompts.append(question)

    num_prompts = len(prompts)
    if num_prompts > model_config["max_batch_size"]:
        raise ValueError("num_prompts={} exceeds max_batch_size={}".format(
            num_prompts, model_config["max_batch_size"]))

    print("  -- Generating completions for num_prompts={} ...".format(
        num_prompts))

    if args.backend == "trt":
        trt_bench(args, model_config, prompts)
    else:
        vllm_bench(args, model_config, prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--backend",
                        type=str,
                        choices=["trt", "vllm"],
                        required=True)
    parser.add_argument("--model-config",
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument("--prompt-scale",
                        type=int,
                        default=None,
                        required=True)

    parser.add_argument("--output-tokens",
                        type=int,
                        default=None,
                        required=True)

    parser.add_argument("--batch-size", type=int, default=None, required=True)

    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--num-bench", type=int, default=4)

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)

    args = parser.parse_args()

    main(args)
