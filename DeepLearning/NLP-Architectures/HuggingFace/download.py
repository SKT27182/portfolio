# script to download a model from huggingface


from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

def download(model_id, save_loc):
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 trust_remote_code=True
                                                 )
    
    tokenizer = AutoTokenizer.from_pretrained(
                                                model_id
                                            )

    # save it
    model.save_pretrained(save_loc)
    tokenizer.save_pretrained(save_loc)

def arg_parse():
    args = argparse.ArgumentParser(add_help=True)

    args.add_argument("--model", type=str, default="tiiuae/falcon-40b-instruct", help="Huggingface model id which we want to download")

    args.add_argument("--save_loc", type=str, default=".", help="Huggingface model id which we want to download")

    args = args.parse_args()

    return args

if __name__ == "__main__":
    print("In")

    args = arg_parse()
    download(args.model, args.save_loc)
