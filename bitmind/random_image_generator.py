from transformers import pipeline
from diffusers import DiffusionPipeline
from transformers import set_seed
from datasets import load_dataset
import bittensor as bt
import torch
import random
import time
import re


class RandomImageGenerator:

    def __init__(
            self
    ):

        #self.prompt_generators = prompt_generators
        #self.image_generators = image_generators

        bt.logging.info("Loading prompt generation models...")
        self.prompt_generator = pipeline(
            'text-generation',
            model='Gustavosta/MagicPrompt-Stable-Diffusion',
            tokenizer='gpt2',
            device=-1)
        with open('./bitmind/data/ideas.txt', 'r') as fin:
            self.ideas_text = fin.readlines()

        bt.logging.info("Loading image generation models...")
        self.diffuser = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16")
        self.diffuser.to("cuda")

    def generate(self, k=1):
        """

        """
        print("Generating prompts...")
        prompts = [
            self.generate_prompt("A realistic image")
            for _ in range(k)
        ]

        print("Generating images...")
        gen_data = []
        for prompt in prompts:
            image_name = f"{time.time()}.jpg"
            gen_image = self.diffuser(prompt=prompt).images[0]
            gen_image.save(image_name)
            gen_data.append({
                'prompt': prompt,
                'image': gen_image,
                'id': image_name
            })

        return gen_data

    def generate_prompt(self, starting_text):
        seed = random.randint(100, 1000000)
        set_seed(seed)

        #response = generator(
        # starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=4)
        response = self.prompt_generator(
            starting_text, max_length=(77 - len(starting_text)), num_return_sequences=1, truncation=True)

        response_list = []
        for x in response:
            resp = x['generated_text'].strip()
            if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
                response_list.append(resp+'\n')

        response_end = "\n".join(response_list)
        response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
        response_end = response_end.replace("<", "").replace(">", "")

        if response_end != "":
            return response_end


