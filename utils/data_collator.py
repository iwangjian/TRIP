# -*- coding: utf-8 -*-
import torch


def max_seq_length(list_l):
    return max(len(l) for l in list_l)

def pad_sequence(list_l, max_len, padding_value=0):
    assert len(list_l) <= max_len
    padding_l = [padding_value] * (max_len - len(list_l))
    padded_list = list_l + padding_l
    return padded_list


class PlanCollator(object):
    """
    Data collator for planning
    """
    def __init__(self, device, padding_idx=0):
        self.device = device
        self.padding_idx = padding_idx
    
    def list_to_tensor(self, list_l):
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=self.padding_idx))
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor

    def varlist_to_tensor(self, list_vl):
        lens = []
        for list_l in list_vl:
            lens.append(max_seq_length(list_l))
        max_len = max(lens)
        
        padded_lists = []
        for list_seqs in list_vl:
            v_list = []
            for list_l in list_seqs:
                v_list.append(pad_sequence(list_l, max_len, padding_value=self.padding_idx))
            padded_lists.append(v_list)
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = data_tensor.masked_fill(data_tensor == self.padding_idx, 0)
        attention_mask = attention_mask.masked_fill(attention_mask != self.padding_idx, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask
    
    def custom_collate(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        batch_ctx_input, batch_ctx_seg = [], []
        batch_target_input, batch_target_seg = [], []
        batch_forward_plan, batch_forward_tgt_mask, batch_forward_neg_plan, batch_forward_neg_mask = [], [], [], []
        batch_backward_plan, batch_backward_tgt_mask, batch_backward_neg_plan, batch_backward_neg_mask = [], [], [], []

        for sample in mini_batch:
            batch_ctx_input.append(sample.context_input_ids)
            batch_ctx_seg.append(sample.context_seg_ids)
            batch_target_input.append(sample.target_input_ids)
            batch_target_seg.append(sample.target_seg_ids)
            
            batch_forward_plan.append(sample.forward_plan_ids)
            batch_forward_tgt_mask.append(sample.forward_tgt_masks)
            batch_forward_neg_plan.append(sample.forward_neg_ids)
            batch_forward_neg_mask.append(sample.forward_neg_masks)
            
            batch_backward_plan.append(sample.backward_plan_ids)
            batch_backward_tgt_mask.append(sample.backward_tgt_masks)
            batch_backward_neg_plan.append(sample.backward_neg_ids)
            batch_backward_neg_mask.append(sample.backward_neg_masks)
        
        # inputs
        ctx_input_ids = self.list_to_tensor(batch_ctx_input)
        ctx_masks = self.get_attention_mask(ctx_input_ids)
        ctx_seg_ids = self.list_to_tensor(batch_ctx_seg)
        tgt_input_ids = self.list_to_tensor(batch_target_input)
        tgt_masks = self.get_attention_mask(tgt_input_ids)
        tgt_seg_ids = self.list_to_tensor(batch_target_seg)
        
        # outputs
        forward_plan_ids = self.list_to_tensor(batch_forward_plan)      # (bsz, seq_len)
        forward_plan_masks = self.get_attention_mask(forward_plan_ids)   # (bsz, seq_len)
        forward_plan_tgt_masks = self.list_to_tensor(batch_forward_tgt_mask)  # (bsz, seq_len)

        forward_neg_ids = self.varlist_to_tensor(batch_forward_neg_plan)   # (bsz, num_seq, seq_len)
        forward_neg_masks = self.get_attention_mask(forward_neg_ids)  # (bsz, num_seq, seq_len)
        forward_neg_tgt_masks = self.varlist_to_tensor(batch_forward_neg_mask)  # (bsz, num_seq, seq_len)
        
        backward_plan_ids = self.list_to_tensor(batch_backward_plan)      # (bsz, seq_len)
        backward_plan_masks = self.get_attention_mask(backward_plan_ids)   # (bsz, seq_len)
        backward_plan_tgt_masks = self.list_to_tensor(batch_backward_tgt_mask)  # (bsz, seq_len)

        backward_neg_ids = self.varlist_to_tensor(batch_backward_neg_plan)   # (bsz, num_seq, seq_len)
        backward_neg_masks = self.get_attention_mask(backward_neg_ids)  # (bsz, num_seq, seq_len)
        backward_neg_tgt_masks = self.varlist_to_tensor(batch_backward_neg_mask)  # (bsz, num_seq, seq_len)
            
        collated_batch = {
            "context": [ctx_input_ids, ctx_masks, ctx_seg_ids],
            "target": [tgt_input_ids, tgt_masks, tgt_seg_ids],
            "forward_plan": [forward_plan_ids, forward_plan_masks, forward_plan_tgt_masks],
            "forward_neg_plan": [forward_neg_ids, forward_neg_masks, forward_neg_tgt_masks],
            "backward_plan": [backward_plan_ids, backward_plan_masks, backward_plan_tgt_masks],
            "backward_neg_plan": [backward_neg_ids, backward_neg_masks, backward_neg_tgt_masks]
        }

        return collated_batch


class DialCollator(object):
    """
    Data collator for dialogue generation
    """
    def __init__(self, device, padding_idx=0):
        self.device = device
        self.padding_idx = padding_idx
    
    def list_to_tensor(self, list_l):
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=self.padding_idx))
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = data_tensor.masked_fill(data_tensor == self.padding_idx, 0)
        attention_mask = attention_mask.masked_fill(attention_mask != self.padding_idx, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask
    
    def custom_collate(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        batch_ctx_input, batch_ctx_seg = [], []
        batch_plan_input, batch_plan_seg = [], []
        
        batch_ctrl = []
        batch_resp = []
        
        for sample in mini_batch:
            batch_ctx_input.append(sample.context_input_ids)
            batch_ctx_seg.append(sample.context_seg_ids)
            batch_plan_input.append(sample.plan_input_ids)
            batch_plan_seg.append(sample.plan_seg_ids)
            
            batch_ctrl.append(sample.control_ids)
            batch_resp.append(sample.response_ids)
        
        # inputs
        ctx_input_ids = self.list_to_tensor(batch_ctx_input)
        ctx_masks = self.get_attention_mask(ctx_input_ids)
        ctx_seg_ids = self.list_to_tensor(batch_ctx_seg)
        plan_input_ids = self.list_to_tensor(batch_plan_input)
        plan_masks = self.get_attention_mask(plan_input_ids)
        plan_seg_ids = self.list_to_tensor(batch_plan_seg)
        
        # outputs
        ctrl_ids = self.list_to_tensor(batch_ctrl)
        ctrl_masks = self.get_attention_mask(ctrl_ids)
        resp_ids = self.list_to_tensor(batch_resp)
        resp_masks = self.get_attention_mask(resp_ids)
        
        collated_batch = {
            "context": [ctx_input_ids, ctx_masks, ctx_seg_ids],
            "plan": [plan_input_ids, plan_masks, plan_seg_ids],
            "control": [ctrl_ids, ctrl_masks],
            "response": [resp_ids, resp_masks]
        }

        return collated_batch


class DialGPT2Collator(object):
    """
    Data collator for dialogue generation
    """
    def __init__(self, device, padding_idx=0):
        self.device = device
        self.padding_idx = padding_idx
    
    def list_to_tensor(self, list_l):
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=self.padding_idx))
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = data_tensor.masked_fill(data_tensor == self.padding_idx, 0)
        attention_mask = attention_mask.masked_fill(attention_mask != self.padding_idx, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask
    
    def custom_collate(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        batch_input, batch_control, batch_label = [], [], []
        
        for sample in mini_batch:
            batch_input.append(sample.input_ids)
            batch_control.append(sample.control_ids)
            batch_label.append(sample.lm_labels)
        
        input_ids = self.list_to_tensor(batch_input)
        control_ids = self.list_to_tensor(batch_control)
        control_masks = self.get_attention_mask(control_ids)
        lm_labels = self.list_to_tensor(batch_label)
        
        collated_batch = {
            "input_ids": input_ids,
            "control_ids": control_ids,
            "control_masks": control_masks,
            "lm_labels": lm_labels
        }

        return collated_batch