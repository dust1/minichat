import argparse
import os
import numpy as np
import tokenizer as tok
from train import ByteTransformer, load_model, sample_logits, _to_numpy, _gpu


def generate(model, prompt, max_gen=300, max_len=512, temperature=1.0, top_k=50,
             top_p=0.95, repetition_penalty=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    generated = tok.encode(prompt)

    for _ in range(max_gen):
        inp = np.array([generated[-max_len:]], dtype=np.int32)
        if _gpu:
            import cupy as cp
            inp = cp.array(inp)

        logits = model.forward(inp, training=False)
        logits_np = _to_numpy(logits[0, -1, :])
        next_id = sample_logits(logits_np, gen_ids=generated, temperature=temperature,
                                top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
        if next_id == tok.EOS_ID:
            break
        generated.append(next_id)
        if len(generated) > max_len:
            generated = generated[-max_len:]

    return tok.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="MiniChat inference (unified tokenizer)")
    parser.add_argument("prompt", type=str, nargs="?", default="从前有座山")
    parser.add_argument("--model-dir", default="./model_weights")
    parser.add_argument("--max-gen", type=int, default=300)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--rep-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Error: model directory '{args.model_dir}' not found.")
        return

    print("Loading model...")
    model = load_model(args.model_dir)

    tok_path = os.path.join(args.model_dir, "bbpe.model")
    if not os.path.isfile(tok_path):
        print(f"Error: tokenizer file '{tok_path}' not found in model directory.")
        return
    tok.load_tokenizer(tok_path)
    if tok.vocab_size() != model.vocab_size:
        print(f"Error: tokenizer vocab size {tok.vocab_size()} != "
              f"model vocab size {model.vocab_size}; checkpoint 与 tokenizer 不匹配")
        return
    print(f"Tokenizer loaded from {tok_path} (vocab={tok.vocab_size()})")

    print("Generating...")
    result = generate(model, args.prompt, max_gen=args.max_gen, max_len=args.max_len,
                      temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                      repetition_penalty=args.rep_penalty, seed=args.seed)

    print()
    print("Prompt   :", args.prompt)
    print("Generated:", result[:500])


if __name__ == "__main__":
    main()
