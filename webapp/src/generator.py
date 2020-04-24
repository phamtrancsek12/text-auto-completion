import re
import torch
import torch.nn.functional as F
from config import *


class Generator:
    """
    Listing Description Generator using GPT-2 model
    """
    def __init__(self):
        self.config = CONFIG_CLASS.from_pretrained(MODEL_PATH)
        self.tokenizer = TOKENIZER_CLASS.from_pretrained(MODEL_PATH)
        self.model = MODEL_CLASS.from_pretrained(MODEL_PATH)

    def generate(self, input, temperature=1, top_p=0.9, length=5, num=3):
        """
        Generate `num` of sample for the given text
        Each sample has `length` tokens
        """
        samples = []
        input_ids = self.tokenizer.encode(input, add_special_tokens=True)
        for _ in range(num):
            output = self.sample_sequence(length, input_ids, temperature=temperature, top_p=top_p)
            output = self.decode_sample(output)
            samples.append(self.format_output(input, output))
        samples = list(set(samples))
        return samples

    @staticmethod
    def format_output(input, output):
        """
        Remove input tokens and limit the return sentences (before delimiter tokens)
        """
        output = output[len(input): ]     # remove input token
        output = " ".join(output.split()) # remove extra space
        output = re.split(SPLIT_REGEX, output)[0] # limit the return text to befor delimiters
        return output

    @staticmethod
    def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def sample_sequence(self, length, context, num_samples=1, temperature=1,
        top_k=0, top_p=0.9, repetition_penalty=1.0, device="cpu"):
        """
        Generate sample sequence
        """
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context

        self.model.eval()
        with torch.no_grad():
            for i in range(length):
                inputs = {"input_ids": generated}
                outputs = self.model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
                next_token_logits = outputs[0][0, -1, :] / (temperature if temperature > 0 else 1.0)

                # reptition penalty from CTRL (https://arxiv.org/abs/1909.05858)
                for _ in set(generated.view(-1).tolist()):
                    next_token_logits[_] /= repetition_penalty

                filtered_logits = Generator.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                if temperature == 0:  # greedy sampling:
                    next_token = torch.argmax(filtered_logits).unsqueeze(0)
                else:
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        return generated

    def decode_sample(self, output):
        """
        Convert model output to tokens
        """
        return self.tokenizer.decode(output[0, 0:].tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True)
