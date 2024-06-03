from transformers import pipeline
from diffusers import DiffusionPipeline
from transformers import set_seed
from datasets import load_dataset
import bittensor as bt
import numpy as np
import torch
import random
import time
import re
import gc

VALID_PROMPT_GENERATOR_NAMES = ['Gustavosta/MagicPrompt-Stable-Diffusion']
VALID_DIFFUSER_NAMES = [
    'stabilityai/stable-diffusion-xl-base-1.0',
    'SG161222/RealVisXL_V4.0'
]

class RandomImageGenerator:

    def __init__(
        self,
        prompt_generator_name='Gustavosta/MagicPrompt-Stable-Diffusion',
        diffuser_name='SG161222/RealVisXL_V4.0',
    ):

        assert prompt_generator_name in VALID_PROMPT_GENERATOR_NAMES, 'invalid prompt generator name'
        assert diffuser_name == 'random' or diffuser_name in VALID_DIFFUSER_NAMES, 'invalid diffuser name'

        self.prompt_generator_name = prompt_generator_name
        self.diffuser_name = diffuser_name

        bt.logging.info(f"Loading prompt generation model ({prompt_generator_name})...")
        self.prompt_generator = pipeline(
            'text-generation',
            model=prompt_generator_name,
            tokenizer='gpt2',
            device=-1)

        if diffuser_name != 'random':
            bt.logging.info(f"Loading image generation model ({diffuser_name})...")
            self.diffuser = DiffusionPipeline.from_pretrained(
                diffuser_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16")
            self.diffuser.to("cuda")
        else:
            bt.logging.info("A random image generation model will be loaded on each generation step.")
            self.diffuser = None


    def generate(self, k=1):
        """

        """
        if self.diffuser_name == 'random':
            self.load_random_diffuser()

        print("Generating prompts...")
        prompts = [
            self.generate_prompt()
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

    def load_random_diffuser(self):
        if self.diffuser is not None:
            bt.logging.info(f"Deleting previous diffuser, freeing memory")
            self.diffuser.to('cpu')
            del self.diffuser
            gc.collect()
            torch.cuda.empty_cache()

        diffuser_name = np.random.choice(VALID_DIFFUSER_NAMES, 1)[0]
        bt.logging.info(f"Loading image generation model ({diffuser_name})...")
        self.diffuser = DiffusionPipeline.from_pretrained(
            diffuser_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16")
        self.diffuser.to("cuda")

    def generate_prompt(self, retry_attempts=10):
        seed = random.randint(100, 1000000)
        set_seed(seed)

        #response = generator(
        # starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=4)

        starters = [
            'A photorealistic portrait',
            'A photorealistic landscape',
            'A photorealistic scene'
        ]

        context = [
            'RAW photo', 'subject', '8k uhd', 'dslr', 'soft lighting', 'high quality', 'film grain', 'Fujifilm XT3'
        ]

        for _ in range(retry_attempts):
            starting_text = np.random.choice(starters, 1)[0]
            response = self.prompt_generator(
                starting_text, max_length=(77 - len(starting_text)), num_return_sequences=1, truncation=True)

            response_list = []
            for x in response:
                resp = x['generated_text'].strip()
                if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
                    response_list.append(resp+'\n')

            prompt = "\n".join(response_list)
            prompt = re.sub('[^ ]+\.[^ ]+','', prompt)
            prompt = prompt.replace("<", "").replace(">", "")

            # temporary removal of extra context (like "featured on artstation") until we've trained our own prompt generator
            prompt = re.split('[,;]', prompt)[0] + ', '
            prompt += ', '.join(np.random.choice(context, np.random.randint(len(context)//2, len(context))))

            if prompt != "":
                return prompt


