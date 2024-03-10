# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from transformers import (
    LogitsProcessorList,
    HammingDiversityLogitsProcessor,
    NoBadWordsLogitsProcessor,
    MinLengthLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    MaxTimeCriteria,
)
from .generation_beam_constraints import PhrasalConstraint
from .generation_beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer


def _get_logits_processor(
    diversity_penalty: float = None,
    repetition_penalty: float = None,
    no_repeat_ngram_size: int = None,
    bad_words_ids: List[List[int]] = None,
    min_length: int = None,
    max_length: int = None,
    eos_token_id: int = None,
    forced_bos_token_id: int = None,
    forced_eos_token_id: int = None,
    num_beams: int = None,
    num_beam_groups: int = None,
    remove_invalid_values: bool = False,
) -> LogitsProcessorList:
    """
    This mathod returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
    :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
    """
    processors = LogitsProcessorList()

    if diversity_penalty is not None and diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
            )
        )
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(
            penalty=repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(
            bad_words_ids, eos_token_id))
    if min_length is not None and eos_token_id is not None and min_length > -1:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
    if forced_eos_token_id is not None:
        processors.append(ForcedEOSTokenLogitsProcessor(
            max_length, forced_eos_token_id))
    if remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    return processors


def _get_stopping_criteria(max_length=None, max_time=None) -> StoppingCriteriaList:
    stopping_criteria = StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    if max_time is not None:
        stopping_criteria.append(MaxTimeCriteria(max_time=max_time))
    return stopping_criteria


def _expand_inputs_for_plan_generation(inputs, expand_size=1):
    ctx_input_ids, ctx_masks, ctx_seg_ids = inputs["context"]
    tgt_input_ids, tgt_masks, tgt_seg_ids = inputs["target"]

    back_plan_ids, back_plan_masks, _ = inputs["backward_plan"]
    for_plan_ids, for_plan_masks, _ = inputs["forward_plan"]

    expanded_return_idx = (
        torch.arange(ctx_input_ids.shape[0]).view(
            -1, 1).repeat(1, expand_size).view(-1).to(ctx_input_ids.device)
    )

    #TODO: simplify code by k-v iterations
    ctx_input_ids = ctx_input_ids.index_select(0, expanded_return_idx)
    ctx_masks = ctx_masks.index_select(0, expanded_return_idx)
    ctx_seg_ids = ctx_seg_ids.index_select(0, expanded_return_idx)

    tgt_input_ids = tgt_input_ids.index_select(0, expanded_return_idx)
    tgt_masks = tgt_masks.index_select(0, expanded_return_idx)
    tgt_seg_ids = tgt_seg_ids.index_select(0, expanded_return_idx)

    back_plan_ids = back_plan_ids.index_select(0, expanded_return_idx)
    back_plan_masks = back_plan_masks.index_select(0, expanded_return_idx)
    for_plan_ids = for_plan_ids.index_select(0, expanded_return_idx)
    for_plan_masks = for_plan_masks.index_select(0, expanded_return_idx)

    expanded_inputs = {
        "context": [ctx_input_ids, ctx_masks, ctx_seg_ids],
        "target": [tgt_input_ids, tgt_masks, tgt_seg_ids],
        "backward_plan": [back_plan_ids, back_plan_masks, None],
        "forward_plan": [for_plan_ids, for_plan_masks, None],
    }

    return expanded_inputs


def _expand_inputs_for_resp_generation(inputs, expand_size=1):
    ctx_input_ids, ctx_masks, ctx_seg_ids = inputs["context"]
    pln_input_ids, pln_masks, pln_seg_ids = inputs["plan"]

    ctrl_ids, ctrl_masks = inputs["control"]
    resp_ids, resp_masks = inputs["response"]

    expanded_return_idx = (
        torch.arange(ctx_input_ids.shape[0]).view(
            -1, 1).repeat(1, expand_size).view(-1).to(ctx_input_ids.device)
    )

    #TODO: simplify code by k-v iterations
    ctx_input_ids = ctx_input_ids.index_select(0, expanded_return_idx)
    ctx_masks = ctx_masks.index_select(0, expanded_return_idx)
    ctx_seg_ids = ctx_seg_ids.index_select(0, expanded_return_idx)

    pln_input_ids = pln_input_ids.index_select(0, expanded_return_idx)
    pln_masks = pln_masks.index_select(0, expanded_return_idx)
    pln_seg_ids = pln_seg_ids.index_select(0, expanded_return_idx)

    ctrl_ids = ctrl_ids.index_select(0, expanded_return_idx)
    ctrl_masks = ctrl_masks.index_select(0, expanded_return_idx)
    resp_ids = resp_ids.index_select(0, expanded_return_idx)
    resp_masks = resp_masks.index_select(0, expanded_return_idx)

    expanded_inputs = {
        "context": [ctx_input_ids, ctx_masks, ctx_seg_ids],
        "plan": [pln_input_ids, pln_masks, pln_seg_ids],
        "control": [ctrl_ids, ctrl_masks],
        "response": [resp_ids, resp_masks]
    }

    return expanded_inputs



def greedy_search(
    model,
    inputs,
    diversity_penalty: float = None,
    repetition_penalty: float = None,
    no_repeat_ngram_size: int = None,
    min_length: int = None,
    max_length: int = None,
    pad_token_id: int = None,
    bos_token_id: int = None,
    eos_token_id: int = None,
    output_scores: bool = False
):
    # prepare distribution pre_processing samplers
    logits_processor = _get_logits_processor(
        diversity_penalty=diversity_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=bos_token_id,
        forced_eos_token_id=eos_token_id,
        num_beams=1,
        num_beam_groups=1
    )

    # prepare stopping criteria
    stopping_criteria = _get_stopping_criteria(
        max_length=max_length, max_time=None)
    
    scores = None
    if output_scores:
        scores = ()

    # prepare input ids
    resp_ids, _ = inputs["response"]
    unfinished_seqs = resp_ids.new(resp_ids.shape[0]).fill_(1)

    # decoding
    while True:
        inputs["response"] = [resp_ids, None]

        model_outputs = model(inputs, is_test=True)

        # compute logits
        next_logits = model_outputs["lm_logits"][:, -1, :]
        next_scores = logits_processor(resp_ids, next_logits)

        # compute next tokens
        next_tokens = torch.argmax(next_scores, dim=-1)
        next_tokens = next_tokens * unfinished_seqs + pad_token_id * (1 - unfinished_seqs)
        resp_ids = torch.cat([resp_ids, next_tokens[:, None]], dim=-1)
        
        # compute unfinished seqs
        unfinished_seqs = unfinished_seqs.mul((next_tokens != eos_token_id).long())

        if unfinished_seqs.max() == 0 or stopping_criteria(resp_ids, scores):
            break

    return resp_ids


def biconstrained_beam_search(
    model,
    inputs,
    num_beams,
    lambda_scale: float = 0.1,
    diversity_penalty: float = None,
    repetition_penalty: float = None,
    no_repeat_ngram_size: int = None,
    min_length: int = None,
    max_length: int = None,
    pad_token_id: int = None,
    bos_token_id: int = None,
    eos_token_id: int = None,
    output_scores: bool = False
):
    # prepare distribution pre_processing samplers
    logits_processor = _get_logits_processor(
        diversity_penalty=diversity_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=bos_token_id,
        forced_eos_token_id=eos_token_id,
        num_beams=num_beams,
        num_beam_groups=1
    )

    # prepare stopping criteria
    stopping_criteria = _get_stopping_criteria(
        max_length=max_length, max_time=None)

    # prepare constraints
    tgt_ids, tgt_masks, _ = inputs["target"]
    back_plan_ids, back_plan_masks, back_plan_tgt_masks = inputs["backward_plan"]
    
    back_plan_ids = torch.cat([back_plan_ids, tgt_ids], dim=1)
    back_plan_masks = torch.cat([back_plan_masks, tgt_masks], dim=1)
    inputs["backward_plan"] = [back_plan_ids, back_plan_masks, back_plan_tgt_masks]

    for_constraints = []
    for word_ids in tgt_ids.cpu().tolist():
        tgt_constraint = PhrasalConstraint(word_ids)
        for_constraints.append(tgt_constraint)
    
    # prepare beam search scorer
    back_const_beam_scorer = BeamSearchScorer(
        batch_size=tgt_ids.size(0),
        num_beams=num_beams,
        device=tgt_ids.device
    )
    for_const_beam_scorer = ConstrainedBeamSearchScorer(
        constraints=for_constraints,
        batch_size=tgt_ids.size(0),
        num_beams=num_beams,
        device=tgt_ids.device,
        length_penalty=0.1
    )

    # interleave input_ids with `num_beams` additional sequences per batch
    model_inputs = _expand_inputs_for_plan_generation(inputs, expand_size=num_beams)
    back_plan_ids, _, _ = model_inputs["backward_plan"]
    for_plan_ids, _, _ = model_inputs["forward_plan"]
    batch_beam_size = back_plan_ids.size(0)
    device = back_plan_ids.device
    batch_size = len(back_const_beam_scorer._beam_hyps)

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    back_beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=device)
    back_beam_scores[:, 1:] = -1e9
    back_beam_scores = back_beam_scores.view((batch_size * num_beams,))
    for_beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=device)
    for_beam_scores[:, 1:] = -1e9
    for_beam_scores = for_beam_scores.view((batch_size * num_beams,))

    back_scores, for_scores = None, None
    back_beam_indices, for_beam_indices = None, None
    if output_scores:
        back_scores, for_scores = (), ()

    while True:
        model_inputs["backward_plan"] = [back_plan_ids, None, None]
        model_inputs["forward_plan"] = [for_plan_ids, None, None]

        model_outputs = model(model_inputs, is_test=True)

        # compute logits
        back_next_logits = model_outputs["lm_backward_logits"][:, -1, :]
        back_next_scores = F.log_softmax(back_next_logits, dim=-1)
        back_next_scores_processed = logits_processor(
            back_plan_ids, back_next_scores)
        back_next_scores = back_next_scores_processed + \
            back_beam_scores[:, None].expand_as(back_next_scores)

        for_next_logits = model_outputs["lm_forward_logits"][:, -1, :]
        for_next_scores = F.log_softmax(for_next_logits, dim=-1)
        for_next_scores_processed = logits_processor(
            for_plan_ids, for_next_scores)
        for_scores_for_all_vocab = for_next_scores_processed.clone()
        for_next_scores = for_next_scores_processed + \
            for_beam_scores[:, None].expand_as(for_next_scores)

        # reshape for beam search
        vocab_size = back_next_scores.shape[-1]

        back_next_scores = back_next_scores.view(
            batch_size, num_beams * vocab_size)
        back_next_scores, back_next_tokens = torch.topk(
            back_next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )
        back_next_indices = (back_next_tokens // vocab_size).long()
        back_next_tokens = back_next_tokens % vocab_size

        for_next_scores = for_next_scores.view(
            batch_size, num_beams * vocab_size)
        for_next_scores, for_next_tokens = torch.topk(
            for_next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )
        for_next_indices = (for_next_tokens // vocab_size).long()
        for_next_tokens = for_next_tokens % vocab_size

        # stateless
        back_beam_outputs = back_const_beam_scorer.process(
            back_plan_ids,
            back_next_scores,
            back_next_tokens,
            back_next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=back_beam_indices
        )
        back_beam_scores = back_beam_outputs["next_beam_scores"]
        back_beam_next_tokens = back_beam_outputs["next_beam_tokens"]

        back_plan_ids = torch.cat(
            [back_plan_ids[back_beam_idx, :], back_beam_next_tokens.unsqueeze(-1)], dim=-1)

        for_beam_outputs = for_const_beam_scorer.process(
            for_plan_ids,
            for_next_scores,
            for_next_tokens,
            for_next_indices,
            for_scores_for_all_vocab,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id
        )
        for_beam_scores = for_beam_outputs["next_beam_scores"]
        for_beam_next_tokens = for_beam_outputs["next_beam_tokens"]
        for_beam_idx = for_beam_outputs["next_beam_indices"]

        for_plan_ids = torch.cat(
            [for_plan_ids[for_beam_idx, :], for_beam_next_tokens.unsqueeze(-1)], dim=-1)

        if back_const_beam_scorer.is_done or stopping_criteria(back_plan_ids, back_scores):
            if for_const_beam_scorer.is_done or stopping_criteria(for_plan_ids, for_scores):
                break

    # run model to obtain hidden states
    model_inputs["backward_plan"] = [back_plan_ids, None, None]
    model_inputs["forward_plan"] = [for_plan_ids, None, None]

    model_outputs = model(model_inputs, is_test=True)

    back_hidden_states = model_outputs["backward_hidden_states"]
    for_hidden_states = model_outputs["forward_hidden_states"]

    # finalize
    back_seq_outputs = back_const_beam_scorer.finalize(
        back_plan_ids,
        back_beam_scores,
        back_next_tokens,
        back_next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=back_beam_indices
    )
    back_cand_seqs = back_seq_outputs["candidate_sequences"]
    back_cand_probs = torch.exp(back_seq_outputs["candidate_sequence_scores"])

    for_seq_outputs = for_const_beam_scorer.finalize(
        for_plan_ids,
        for_beam_scores,
        for_next_tokens,
        for_next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length
    )
    for_cand_seqs = for_seq_outputs["candidate_sequences"]

    # prepare output masks
    back_seq_len = back_cand_seqs.size(-1) - 1
    back_hidden_states = back_hidden_states[:, :back_seq_len, :].reshape(batch_size, num_beams, back_seq_len, -1)
    back_cand_masks = back_cand_seqs.masked_fill(back_cand_seqs < 100, 0)
    back_cand_masks = back_cand_masks.masked_fill(back_cand_masks >= 100, 1)
    back_cand_masks = back_cand_masks[:, :, 1:]

    for_seq_len = for_cand_seqs.size(-1) - 1
    for_hidden_states = for_hidden_states[:, :for_seq_len, :].reshape(batch_size, num_beams, for_seq_len, -1)
    for_cand_masks = for_cand_seqs.masked_fill(for_cand_seqs < 100, 0)
    for_cand_masks = for_cand_masks.masked_fill(for_cand_masks >= 100, 1)
    for_cand_masks = for_cand_masks[:, :, 1:]

    # select the best hypotheses with birectional agreement
    best_back_seqs = back_cand_seqs.new(batch_size, back_seq_len+1).fill_(pad_token_id)

    for idx in range(batch_size):
        for jdx_back in range(num_beams):
            dist_scores = []
            for jdx_for in range(num_beams):
                back_seq_state = back_hidden_states[idx, jdx_back, :, :].unsqueeze(0)  # (1, back_seq_len, hidden_size)
                back_seq_mask = back_cand_masks[idx, jdx_back, :].unsqueeze(0)         # (1, back_seq_len)
                for_seq_state = for_hidden_states[idx, jdx_for, :, :].unsqueeze(0)
                for_seq_mask = for_cand_masks[idx, jdx_for, :].unsqueeze(0)
                dscore = model.compute_bi_distance(
                    back_seq_state,
                    for_seq_state,
                    back_seq_mask,
                    for_seq_mask
                )
                dist_scores.append(float(dscore))
            avg_dscore = np.mean(dist_scores)
            back_cand_probs[idx, jdx_back] += (-lambda_scale * avg_dscore)
       
        best_index = torch.argmax(back_cand_probs[idx, :])
        best_seq = back_cand_seqs[idx, best_index, :]
        best_back_seqs[idx, :len(best_seq)] = best_seq

    return best_back_seqs


def beam_search(
    model,
    inputs,
    num_beams,
    diversity_penalty: float = None,
    repetition_penalty: float = None,
    no_repeat_ngram_size: int = None,
    min_length: int = None,
    max_length: int = None,
    pad_token_id: int = None,
    bos_token_id: int = None,
    eos_token_id: int = None,
    output_scores: bool = False
):
    # prepare distribution pre_processing samplers
    logits_processor = _get_logits_processor(
        diversity_penalty=diversity_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=bos_token_id,
        forced_eos_token_id=eos_token_id,
        num_beams=num_beams,
        num_beam_groups=1
    )

    # prepare stopping criteria
    stopping_criteria = _get_stopping_criteria(
        max_length=max_length, max_time=None)
    
    ctx_input_ids, _, _ = inputs["context"]
    beam_scorer = BeamSearchScorer(
        batch_size=ctx_input_ids.size(0),
        num_beams=num_beams,
        device=ctx_input_ids.device
    )
    
    # interleave input_ids with `num_beams` additional sequences per batch
    model_inputs = _expand_inputs_for_resp_generation(inputs, expand_size=num_beams)

    resp_ids, _ = model_inputs["response"]
    
    batch_beam_size = resp_ids.size(0)
    device = resp_ids.device
    batch_size = len(beam_scorer._beam_hyps)

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    scores = None
    beam_indices = None
    if output_scores:
        scores = ()

    while True:
        model_inputs["response"] = [resp_ids, None]

        model_outputs = model(model_inputs, is_test=True)

        # compute logits
        next_logits = model_outputs["lm_logits"][:, -1, :]
        
        # (batch_size * num_beams, vocab_size)
        next_scores = F.log_softmax(next_logits, dim=-1)
        next_scores_processed = logits_processor(resp_ids, next_scores)
        next_scores = next_scores_processed + \
            beam_scores[:, None].expand_as(next_scores)

        # reshape for beam search
        vocab_size = next_scores.shape[-1]

        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size)
        next_scores, next_tokens = torch.topk(
            next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )
        next_indices = (next_tokens // vocab_size).long()
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            resp_ids,
            next_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        resp_ids = torch.cat(
            [resp_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        if beam_scorer.is_done or stopping_criteria(resp_ids, scores):
            break

    seq_outputs = beam_scorer.finalize(
        resp_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices
    )
    best_seqs = seq_outputs["sequences"]

    return best_seqs
