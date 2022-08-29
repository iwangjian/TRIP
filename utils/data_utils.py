# -*- coding: utf-8 -*-UNK
from transformers import BertTokenizer, GPT2Tokenizer

PAD = "[PAD]"     # consistent with Bert tokenizer
UNK = "[UNK]"     # consistent with Bert tokenizer
SEP = "[SEP]"     # consistent with Bert tokenizer

ACT = "[A]"       # denote an action
TPC = "[T]"       # denote a topic
BOS = "[BOS]"     # begin of sequence
EOS = "[EOS]"     # end of sequence

IGNORE_INDEX = -100    # default in CrossEntropyLoss


def get_tokenizer(config_dir, name="bert"):
    if name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(config_dir)
        special_token_map = {"additional_special_tokens": [ACT, TPC, SEP, PAD]}
        num_added_tokens = tokenizer.add_special_tokens(special_token_map)
        special_token_id_dict = {
            "pad_token_id": tokenizer.convert_tokens_to_ids(PAD),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    else:
        tokenizer = BertTokenizer.from_pretrained(config_dir)
        special_token_map = {"additional_special_tokens": [ACT, TPC, BOS, EOS]}
        num_added_tokens = tokenizer.add_special_tokens(special_token_map)
        special_token_id_dict = {
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.convert_tokens_to_ids(BOS),
            "eos_token_id": tokenizer.convert_tokens_to_ids(EOS),
        }
    return tokenizer, num_added_tokens, special_token_id_dict


def convert_ids_to_tokens(output, tokenizer, lang="en"):
    sentences = []
    for idx in range(output.size(0)):
        decode_tokens = tokenizer.decode(output[idx, :]).split()
        return_tokens = []
        for token in decode_tokens:
            if token == BOS:
                continue
            elif token == EOS or token == PAD:
                break
            elif token.startswith(EOS):
                break
            elif token.endswith(EOS):
                return_tokens.append(token.replace(EOS, ""))
                break
            elif token.endswith("<|endoftext|>"):
                return_tokens.append(token.replace("<|endoftext|>", ""))
                break
            elif token.upper() == "NULL":
                return_tokens.append("NULL")
            else:
                return_tokens.append(token)
        if lang == "zh":
            return_str = "".join(return_tokens)
        else:
            return_str = " ".join(return_tokens)
        sentences.append(return_str)
    return sentences


def get_eval_output(path_str, backward=True):
    if backward:
        # parse backward path
        # i.e., [A]...[T]...[A]act[T]tpc
        try:
            eval_span = path_str.split(ACT)[-1].strip()
        except IndexError:
            eval_span = None
        if eval_span is None:
            action, topic = UNK, UNK
        else:
            try:
                action = eval_span.split(TPC)[0].strip()
            except IndexError:
                action = UNK
            try:
                topic = eval_span.split(TPC)[-1].strip()
            except IndexError:
                topic = UNK
    else:
        # parse forward path
        # i.e., [A]act[T]tpc[A]...[T]...
        try:
            action = path_str.split(TPC)[0].split(ACT)[-1].strip()
        except IndexError:
            action = UNK
        try:
            if path_str.startswith(ACT):
                topic = path_str.split(ACT)[1].split(TPC)[-1].strip()
            else:
                topic = path_str.split(ACT)[0].split(TPC)[-1].strip()
        except IndexError:
            topic = UNK
    return (action, topic)

def get_plan_path(path_str, backward=True):
    if backward:
        plan_list = list(path_str.split(ACT))
        plan_list.reverse()
        plan_path = ACT.join(plan_list)
        if plan_path.endswith(ACT):
            plan_path = plan_path[:-3]
        if not plan_path.startswith(ACT):
            plan_path = ACT + plan_path
    else:
        plan_path = path_str
    return plan_path
    