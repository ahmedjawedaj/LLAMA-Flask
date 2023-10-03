import sys
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from flask import Flask, request, make_response, jsonify

from lit_llama import LLaMA, Tokenizer, as_8_bit_quantized

app = Flask(__name__)

llama_model = None


def compile_model():
    global llama_model
    if llama_model:
        return
    llama_model = LLAMA()


def get_llama_model():
    compile_model()
    return llama_model

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    The implementation of this function is modified from A. Karpathy's nanoGPT.
    Args:
        model: The model to use.
        idx: Tensor of shape (B, T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    B, T = idx.shape
    T_new = T + max_new_tokens
    empty = torch.empty(B, T_new, dtype=idx.dtype, device=idx.device)
    empty[:, :T] = idx
    idx = empty

    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:, :t]
        # if the sequence context is growing too long we must crop it at max_seq_length
        idx_cond = idx_cond if T <= max_seq_length else idx_cond[:, -max_seq_length:]

        # forward
        logits = model(idx_cond)
        logits = logits[:, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # concatenate the new column
        idx[:, t:] = idx_next

    return idx


class LLAMA:

    """Generates text samples based on a pre-trained LLaMA model and tokenizer.
    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        # compile: Whether to compile the model.
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model using the `LLM.int8()` method
    """

    def __init__(self):
        self.num_samples: int = 1
        self.max_new_tokens: int = 50
        self.top_k: int = 200
        self.temperature: float = 0.8
        # compilation fails as it does not support torch.complex64 for RoPE
        # compile: bool = False
        self.accelerator: str = "auto"
        self.checkpoint_path: Optional[Path] = None
        self.tokenizer_path: Optional[Path] = None
        self.model_size: str = "7B"
        self.quantize: bool = False
        self.tokenizer = None
        self.fabric = None

    def load_model(self):
        if not self.checkpoint_path:
            self.checkpoint_path = Path(f"./checkpoints/lit-llama/{self.model_size}/state_dict.pth")
        if not self.tokenizer_path:
            self.tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
        assert self.checkpoint_path.is_file()
        assert self.tokenizer_path.is_file()

        self.fabric = L.Fabric(accelerator=self.accelerator, devices=1)

        with as_8_bit_quantized(self.fabric.device, enabled=self.quantize):
            print("Loading model ...", file=sys.stderr)
            t0 = time.time()
            model = LLaMA.from_name(self.model_size)
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint)
            print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

        model.eval()

        # if compile:
        #     model = torch.compile(model)

        self.model = self.fabric.setup_module(model)

        self.tokenizer = Tokenizer(self.tokenizer_path)

    def generate_text(self, prompt, max_new_tokens, temperature, top_k):

        encoded_prompt = self.tokenizer.encode(prompt, bos=True, eos=False, max_length=self.max_new_tokens, device=self.fabric.device)
        encoded_prompt = encoded_prompt[None, :]  # add batch dimension

        L.seed_everything(1234)
        t0 = time.perf_counter()

        output = ""
        for _ in range(self.num_samples):
            y = generate(
                self.model,
                encoded_prompt,
                max_new_tokens,
                self.model.config.block_size,  # type: ignore[union-attr,arg-type]
                temperature=temperature,
                top_k=top_k,
            )[0]  # unpack batch dimension
            output = self.tokenizer.decode(y)
            print(output)

        t = time.perf_counter() - t0
        print(f"\n\nTime for inference: {t:.02f} sec total, {self.num_samples * max_new_tokens / t:.02f} tokens/sec",
              file=sys.stderr)
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
        generated = {"text": output, "generation_time": f"{t:06}s"}
        return generated


def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/complete', methods=['POST', 'OPTIONS'])
def complete():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST":  # The actual request following the preflight
        content = request.json
        print("========json content ====", content)

        prompt = content["text"]
        top_p = float(content["top_p"])
        top_k = int(content["top_k"])
        temp = float(content["temperature"])
        length = int(content["length"])

        response = MODEL_API.generate_text(prompt, length, temp, top_k)

        print("adding queue response ===========")
        try:
            return _corsify_actual_response(jsonify({"completion": response}))
        except Exception as e:
            return jsonify({"completion":{"generation_time":"0.8679995536804199s","text":["I am not feeling well, We can talk later."]}})
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))


MODEL_API = get_llama_model()
MODEL_API.load_model()

if __name__ == "__main__":
    app.run(port='5000', host="0.0.0.0", debug=False)
