# -*- coding: utf-8 -*-
import logging
import os
import json
import pickle
import random
import dataclasses
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import SEP, ACT, TPC, BOS, EOS, IGNORE_INDEX


@dataclass(frozen=True)
class InputFeature:
    """
    A single set of features of data for planning.
    Property names are the same names as the corresponding inputs to a model.
    """
    context_input_ids: List[int]
    context_seg_ids: List[int]
    target_input_ids: List[int]
    target_seg_ids: List[int]
    
    forward_plan_ids: List[int]
    forward_tgt_masks: List[int]
    forward_neg_ids: List[List[int]]
    forward_neg_masks: List[List[int]]
    
    backward_plan_ids: List[int]
    backward_tgt_masks: List[int]
    backward_neg_ids: List[List[int]]
    backward_neg_masks: List[List[int]]

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass(frozen=True)
class DialGPT2InputFeature:
    """
    A single set of features of data for dialogue generation.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    control_ids: List[int]
    lm_labels: List[int]

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"
        

class PlanDataset(Dataset):
    """
    Self-defined PlanDataset class for planning.
    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, 
        data_path,  
        data_partition,
        tokenizer,
        cache_dir=None, 
        max_seq_len=512,
        turn_type_size=16,
        num_negatives=3,
        is_test=False
    ):
        self.data_partition = data_partition
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.max_seq_len = max_seq_len
        self.turn_type_size = turn_type_size
        self.num_negatives = num_negatives
        self.is_test = is_test
        
        self.instances = []
        self._cache_instances(data_path)

    def _cache_instances(self, data_path):
        """
        Load data tensors into memory or create the dataset when it does not exist.
        """
        signature = "{}_cache.pkl".format(self.data_partition)
        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, signature)
        else:
            cache_dir = os.mkdir("caches")
            cache_path = os.path.join(cache_dir, signature)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                logging.info("Loading cached instances from {}".format(cache_path))
                self.instances = pickle.load(f)
        else:          
            logging.info("Loading raw data from {}".format(data_path))
            all_samples = []
            with open(data_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    sample = json.loads(line.strip())
                    if self.is_test:
                        data_sample = {
                            "user_profile": sample["user_profile"],
                            "knowledge": sample["knowledge"],
                            "conversation": sample["conversation"],
                            "target": sample["target"]
                        }
                    else:    
                        data_sample = {
                            "user_profile": sample["user_profile"],
                            "knowledge": sample["knowledge"],
                            "conversation": sample["conversation"],
                            "target": sample["target"],
                            "action_path": sample["action_path"],   # for action planning
                            "topic_path": sample["topic_path"]      # for topic planning
                        }
                    all_samples.append(data_sample)
            
            logging.info("Creating cache instances {}".format(signature))
            for sample in tqdm(all_samples):
                ctx_ids, ctx_segs = self._parse_input_context(sample)
                tgt_ids, tgt_segs = self._parse_input_target(sample)
                forward_plan_ids, forward_tgt_masks, forward_neg_ids, forward_neg_masks, \
                    backward_plan_ids, backward_tgt_masks, backward_neg_ids, backward_neg_masks = self._parse_output_plan(sample)
                
                inputs = {
                    "context_input_ids": ctx_ids,
                    "context_seg_ids": ctx_segs,
                    "target_input_ids": tgt_ids,
                    "target_seg_ids": tgt_segs,
                    "forward_plan_ids": forward_plan_ids,
                    "forward_tgt_masks": forward_tgt_masks,
                    "forward_neg_ids": forward_neg_ids,
                    "forward_neg_masks": forward_neg_masks,
                    "backward_plan_ids": backward_plan_ids,
                    "backward_tgt_masks": backward_tgt_masks,
                    "backward_neg_ids": backward_neg_ids,
                    "backward_neg_masks": backward_neg_masks
                }
                feature = InputFeature(**inputs)
                self.instances.append(feature)

            with open(cache_path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))


    def _parse_input_context(self, sample: dict):
        # user profile
        profile_tokens, profile_segs = [], []
        for k, v in sample["user_profile"].items():
            k_toks = self.tokenizer.tokenize(k)
            v_toks = self.tokenizer.tokenize(v)
            profile_tokens = profile_tokens + k_toks + v_toks
            profile_segs = profile_segs + len(k_toks)*[0] + len(v_toks)*[0]
        profile_tokens = profile_tokens + [SEP]
        profile_segs = profile_segs + [1]
        #print("profile_tokens: ", profile_tokens)

        # domain knowledge
        kg_tokens, kg_segs = [], []
        for kg in sample["knowledge"]:
            kg_tok = self.tokenizer.tokenize(" ".join(kg))
            kg_tokens = kg_tokens + kg_tok + [SEP]
            kg_segs = kg_segs + len(kg_tok)*[0] + [1]
        #print("kg_tokens: ", kg_tokens)

        # dialogue history
        conv_tokens, conv_segs = [], []
        history = sample["conversation"]
        if len(history) > self.turn_type_size:
            history = history[-self.turn_type_size:]
        for h in history:
            h_toks = self.tokenizer.tokenize(h)
            conv_tokens = conv_tokens + h_toks
            conv_segs = conv_segs + len(h_toks)*[0]
        conv_tokens = conv_tokens + [SEP]
        conv_segs = conv_segs + [1]
        #print("conv_tokens: ", conv_tokens)
        #print("")
        
        # concat as context
        ctx_tokens =  conv_tokens + kg_tokens + profile_tokens
        ctx_segs = conv_segs + kg_segs + profile_segs
        ctx_ids = self.tokenizer.convert_tokens_to_ids(ctx_tokens)
        assert len(ctx_ids) == len(ctx_segs)
        
        if len(ctx_ids) > self.max_seq_len:
            ctx_ids = ctx_ids[:self.max_seq_len]
            ctx_segs = ctx_segs[:self.max_seq_len]
        
        return (ctx_ids, ctx_segs)
    
    def _parse_input_target(self, sample: dict):
        # target
        act_toks = self.tokenizer.tokenize(sample["target"][0])
        tpc_toks = self.tokenizer.tokenize(sample["target"][1])
        target_tokens = [ACT] + act_toks + [TPC] + tpc_toks
        target_segs = [1] + [0]*len(act_toks) + [1] + [0]*len(tpc_toks)
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        assert len(target_ids) == len(target_segs)
        
        return (target_ids, target_segs)
    
    def _get_negative_plan(self, plan_str, target_topic, knowledge_triples):
        assert target_topic in plan_str
        candidates = set()
        path_topics = set()
        for spo in knowledge_triples:
            subj, rel, obj = spo
            if subj != target_topic:
                if subj in plan_str:
                    path_topics.add(subj)
                else:
                    candidates.add(subj)
            # TODO: revise negative sampling
            if obj != target_topic:
                if obj in plan_str:
                    path_topics.add(obj)
                else:
                    candidates.add(obj)
            
        if len(candidates) < self.num_negatives:
            candidates = candidates | path_topics
        assert len(candidates) >= self.num_negatives

        neg_targets = random.sample(list(candidates), k=self.num_negatives)
        neg_plans = []
        for neg_t in neg_targets:
            neg_p = plan_str.replace(target_topic, neg_t)
            neg_plans.append(neg_p)
        return neg_plans

    def _parse_output_plan(self, sample: dict):
        bos_token_id = self.tokenizer.convert_tokens_to_ids(BOS)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(EOS)
        
        if self.is_test:    
            forward_plan_ids = [bos_token_id]
            forward_tgt_masks = [0]
            forward_neg_ids = [[bos_token_id] * self.num_negatives]
            forward_neg_masks = [[0] * self.num_negatives]

            backward_plan_ids = [bos_token_id]
            backward_tgt_masks = [0]
            backward_neg_ids = [[bos_token_id] * self.num_negatives]
            backward_neg_masks = [[0] * self.num_negatives]
        else:
            assert "action_path" in sample and "topic_path" in sample
            # forward path
            forward_plan = self._get_forward_path(
                sample["action_path"], sample["topic_path"],
                sample["target"][0], sample["target"][1]
            )
            forward_plan_tokens = self.tokenizer.tokenize(forward_plan)
            forward_plan_ids = [bos_token_id] + self.tokenizer.convert_tokens_to_ids(forward_plan_tokens) + [eos_token_id]
            forward_tgt_masks = [0] + self._get_target_position_mask(forward_plan_tokens, backward=False) + [0] 
            # negative forward paths
            forward_neg_plan = self._get_negative_plan(forward_plan, sample["target"][1], sample["knowledge"])
            forward_neg_tokens = [self.tokenizer.tokenize(neg) for neg in forward_neg_plan]
            forward_neg_ids = [[bos_token_id] + self.tokenizer.convert_tokens_to_ids(toks) + [eos_token_id] \
                for toks in forward_neg_tokens]
            forward_neg_masks = [[0] + self._get_target_position_mask(toks, backward=False) + [0] \
                for toks in forward_neg_tokens]
            for ids, masks in zip(forward_neg_ids, forward_neg_masks):
                assert len(ids) == len(masks)
            '''
            print("forward_plan_tokens: ", forward_plan_tokens)
            print("forward_plan_ids: ", forward_plan_ids)
            print("forward_tgt_masks: ", forward_tgt_masks)
            
            print("forward_neg_tokens: ")
            for fnt in forward_neg_tokens:
                print(fnt)
            print("forward_neg_ids: ")
            for fni in forward_neg_ids:
                print(fni)
            print("forward_neg_masks: ")
            for fnm in forward_neg_masks:
                print(fnm)
            '''
            # backward path
            backward_plan = self._get_backward_path(
                sample["action_path"], sample["topic_path"],
                sample["target"][0], sample["target"][1]
            )
            backward_plan_tokens = self.tokenizer.tokenize(backward_plan)
            backward_plan_ids = [bos_token_id] + self.tokenizer.convert_tokens_to_ids(backward_plan_tokens) + [eos_token_id]
            backward_tgt_masks = [0] + self._get_target_position_mask(backward_plan_tokens, backward=True) + [0]
            # negative backward paths
            backward_neg_plan = self._get_negative_plan(backward_plan, sample["target"][1], sample["knowledge"])
            backward_neg_tokens = [self.tokenizer.tokenize(neg) for neg in backward_neg_plan]
            backward_neg_ids = [[bos_token_id] + self.tokenizer.convert_tokens_to_ids(toks) + [eos_token_id] \
                for toks in backward_neg_tokens]
            backward_neg_masks = [[0] + self._get_target_position_mask(toks, backward=True) + [0] \
                for toks in backward_neg_tokens]
            for ids, masks in zip(backward_neg_ids, backward_neg_masks):
                assert len(ids) == len(masks)
            '''
            print("backward_plan_tokens: ", backward_plan_tokens)
            print("backward_plan_ids: ", backward_plan_ids)
            print("backward_tgt_masks: ", backward_tgt_masks)
            
            print("backward_neg_tokens: ")
            for bnt in backward_neg_tokens:
                print(bnt)
            print("backward_neg_ids: ")
            for bni in backward_neg_ids:
                print(bni)
            print("backward_neg_masks: ")
            for bnm in backward_neg_masks:
                print(bnm)
            '''
        
        return (forward_plan_ids, forward_tgt_masks, forward_neg_ids, forward_neg_masks, \
            backward_plan_ids, backward_tgt_masks, backward_neg_ids, backward_neg_masks)

    @staticmethod
    def _get_forward_path(action_path: list, topic_path: list, target_action: str, target_topic: str):
        ptr = -1
        for idx in range(len(action_path)):
            if action_path[idx] == target_action and topic_path[idx] == target_topic:
                ptr = idx
                break
        if ptr > 0:
            action_path = action_path[:ptr+1]
            topic_path = topic_path[:ptr+1]
        elif ptr == 0:
            action_path = [action_path[0]]
            topic_path = [topic_path[0]]
        else:
            action_path = action_path + [target_action]
            topic_path = topic_path + [target_topic]
        path_str = ""
        for a, t in zip(action_path, topic_path):
            if not t in path_str:
                path_str += "%s%s%s%s" % (ACT, a, TPC, t)
            elif not a in path_str:
                path_str += "%s%s%s%s" % (ACT, a, TPC, t)
        return path_str
    
    @staticmethod
    def _get_backward_path(action_path: list, topic_path: list, target_action: str, target_topic: str):
        ptr = -1
        for idx in range(len(action_path)):
            if action_path[idx] == target_action and topic_path[idx] == target_topic:
                ptr = idx
                break
        if ptr > 0:
            action_path = action_path[:ptr+1]
            topic_path = topic_path[:ptr+1]
            action_path.reverse()
            topic_path.reverse()
        elif ptr == 0:
            action_path = [action_path[0]]
            topic_path = [topic_path[0]]
        else:
            action_path = [target_action] + action_path
            topic_path = [target_topic] + topic_path
        path_str = ""
        for a, t in zip(action_path, topic_path):
            if not t in path_str:
                path_str += "%s%s%s%s" % (ACT, a, TPC, t)
            elif not a in path_str:
                path_str += "%s%s%s%s" % (ACT, a, TPC, t)
        return path_str
    
    @staticmethod
    def _get_target_position_mask(path_tokens: list, backward: bool = False):
        position_mask = [0] * len(path_tokens)
        if backward:
            end_idx, end_act = -1, -1
            for idx, tok in enumerate(path_tokens):
                if tok == ACT:
                    end_act += 1
                if end_act == 1:
                    end_idx = idx
                    break
            if end_idx == -1:
                end_idx = len(path_tokens)
            for idx in range(0, end_idx):
                position_mask[idx] = 1   # should be computed for contrastive generation
        else:
            start_idx = len(path_tokens)
            while start_idx >= 0:
                start_idx -= 1
                if path_tokens[start_idx] == ACT:
                    break
            for idx in range(start_idx, len(path_tokens)):
                position_mask[idx] = 1   # should be computed for contrastive generation
        return position_mask

    @staticmethod
    def _read_vocab(file_path):
        vocab = []
        with open(file_path, 'r', encoding='utf-8') as fr:
            for word in fr:
                vocab.append(word.strip())
        return vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]


class DialGPT2Dataset(Dataset):
    """
    Self-defined DialGPT2Dataset class for dialogue generation.
    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, 
        data_path,  
        data_partition,
        tokenizer,
        special_tokens_dict,
        cache_dir=None, 
        max_seq_len=512,
        turn_type_size=16,
        is_test=False,
        plan_path=None,
        lower_case=False
    ):
        self.data_partition = data_partition
        self.tokenizer = tokenizer
        self.special_tokens_dict = special_tokens_dict
        self.cache_dir = cache_dir
        self.max_seq_len = max_seq_len
        self.turn_type_size = turn_type_size
        self.is_test = is_test
        self.lower_case = lower_case
        
        self.instances = []
        self._cache_instances(data_path, plan_path)

    def _cache_instances(self, data_path, plan_path=None):
        """
        Load data tensors into memory or create the dataset when it does not exist.
        """
        signature = "{}_dial.pkl".format(self.data_partition)
        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, signature)
        else:
            cache_dir = os.mkdir("caches")
            cache_path = os.path.join(cache_dir, signature)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                logging.info("Loading cached instances from {}".format(cache_path))
                self.instances = pickle.load(f)
        else:          
            if self.is_test:
                if plan_path is None:
                    raise ValueError("`plan_path` should not be None during inference!")
                
                logging.info("Loading raw data from {}".format(plan_path))
                all_plans = []
                with open(plan_path, 'r', encoding='utf-8') as fr:
                    for line in fr:
                        plan = json.loads(line.strip())
                        all_plans.append(plan)
                
                logging.info("Loading raw data from {}".format(data_path))
                all_samples = []
                with open(data_path, 'r', encoding='utf-8') as fp:
                    for line in fp:
                        sample = json.loads(line.strip())
                        data_sample = {
                            "user_profile": sample["user_profile"],
                            "knowledge": sample["knowledge"],
                            "conversation": sample["conversation"]
                        }
                        all_samples.append(data_sample)
                
                assert len(all_samples) == len(all_plans)
                for sample, plan in zip(all_samples, all_plans):
                    sample["plan_path"] = plan["plan_path"]
            
            else:
                logging.info("Loading raw data from {}".format(data_path))
                all_samples = []
                with open(data_path, 'r', encoding='utf-8') as fp:
                    for line in fp:
                        sample = json.loads(line.strip())    
                        data_sample = {
                            "user_profile": sample["user_profile"],
                            "knowledge": sample["knowledge"],
                            "conversation": sample["conversation"],
                            "target": sample["target"],
                            "action_path": sample["action_path"],
                            "topic_path": sample["topic_path"],
                            "response": sample["response"]
                        }
                        all_samples.append(data_sample)
            
            logging.info("Creating cache instances {}".format(signature))
            for sample in tqdm(all_samples):
                input_ids, control_ids, lm_labels = self._parse_sample(sample)
                inputs = {
                    "input_ids": input_ids,
                    "control_ids": control_ids,
                    "lm_labels": lm_labels
                }
                feature = DialGPT2InputFeature(**inputs)
                self.instances.append(feature)

            with open(cache_path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))
    
    @staticmethod
    def _get_plan_path(action_path: list, topic_path: list, target_action: str, target_topic: str, lower_case: bool=False):
        ptr = -1
        for idx in range(len(action_path)):
            if action_path[idx] == target_action and topic_path[idx] == target_topic:
                ptr = idx
                break
        if ptr > 0:
            action_path = action_path[:ptr+1]
            topic_path = topic_path[:ptr+1]
        elif ptr == 0:
            action_path = [action_path[0]]
            topic_path = [topic_path[0]]
        else:
            action_path = action_path + [target_action]
            topic_path = topic_path + [target_topic]
        path_str = ""
        for a, t in zip(action_path, topic_path):
            if lower_case:
                a = a.lower()
                t = t.lower()
            if not t in path_str:
                path_str += "%s%s%s%s" % (ACT, a, TPC, t)
            elif not a in path_str:
                path_str += "%s%s%s%s" % (ACT, a, TPC, t)
        return path_str

    def _parse_sample(self, sample: dict):
        bos_token_id = self.special_tokens_dict["bos_token_id"]
        eos_token_id = self.special_tokens_dict["eos_token_id"]
        
        # user profile
        profile_tokens = []
        for k, v in sample["user_profile"].items():
            if self.lower_case:
                k = k.lower()
                v = v.lower()
            k_toks = self.tokenizer.tokenize(k)
            v_toks = self.tokenizer.tokenize(v)
            profile_tokens +=  k_toks + v_toks + [SEP]

        # domain knowledge
        kg_tokens = []
        for kg in sample["knowledge"]:
            kg_str = " ".join(kg)
            if self.lower_case:
                kg_str = kg_str.lower()
            kg_toks = self.tokenizer.tokenize(kg_str)
            kg_tokens +=  kg_toks + [SEP]
        
        # plan path
        if self.is_test:
            assert "plan_path" in sample
            plan_path = sample["plan_path"]
        else:
            assert "action_path" in sample and "topic_path" in sample
            plan_path = self._get_plan_path(
                sample["action_path"], sample["topic_path"],
                sample["target"][0], sample["target"][1],
                lower_case=self.lower_case
            )
        plan_tokens = self.tokenizer.tokenize(plan_path)
        control_ids = [bos_token_id] + self.tokenizer.convert_tokens_to_ids(plan_tokens) + [eos_token_id]
        #print("plan_path: ", plan_path)
        #print("plan_tokens: ", plan_tokens)

        # dialogue history
        conv_tokens = []
        history = sample["conversation"]
        if len(history) > self.turn_type_size:
            history = history[-self.turn_type_size:]
        for h in history:
            if self.lower_case:
                h = h.lower()
            h_toks = self.tokenizer.tokenize(h)
            conv_tokens += h_toks + [SEP]
        
        # concat as context
        ctx_tokens =  profile_tokens + kg_tokens + plan_tokens + [SEP] + conv_tokens
        ctx_ids = self.tokenizer.convert_tokens_to_ids(ctx_tokens)
        if len(ctx_ids) > self.max_seq_len - 1:
            ctx_tokens = ctx_tokens[-self.max_seq_len+1:]
            ctx_ids = ctx_ids[-self.max_seq_len+1:]
        #print("ctx_tokens: ", ctx_tokens)
        #print("ctx_ids: ", ctx_ids)
        #print("")

        if self.is_test:    
            input_ids = [bos_token_id] + ctx_ids
            lm_labels = [IGNORE_INDEX] * (len(ctx_ids) + 1)
        else:
            resp_str = sample["response"].lower() if self.lower_case else sample["response"]
            resp_tokens = self.tokenizer.tokenize(resp_str)
            resp_ids = self.tokenizer.convert_tokens_to_ids(resp_tokens) + [eos_token_id]
            #print("resp_tokens: ", resp_tokens)
            #print("resp_ids: ", resp_ids)
            #print("")
            input_ids = [bos_token_id] + ctx_ids + resp_ids
            lm_labels = [IGNORE_INDEX] * (len(ctx_ids) + 1) + resp_ids
        
        return (input_ids, control_ids, lm_labels)
  
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]
