# -*- coding: utf-8 -*-
import torch
from torch import nn
from transformers import BertModel
from model.nn_decoder import TransformerDecoder
from model.generation_utils import biconstrained_beam_search


class TRIP(nn.Module):
    """
    Model class: TRIP
    Args:
        args: All necessary arguments for the model.
    """
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.pad_token_id = args.pad_token_id
        self.bos_token_id = args.bos_token_id
        self.eos_token_id = args.eos_token_id
        self.beta = args.beta
        self.gamma = args.gamma
        self.tau = args.tau

        self.context_encoder = BertModel.from_pretrained(args.bert_dir)
        self.context_encoder.resize_token_embeddings(args.vocab_size)

        self.target_encoder = BertModel.from_pretrained(args.bert_dir)
        self.target_encoder.resize_token_embeddings(args.vocab_size)
      
        args_decoder = {
            "hidden_size": args.hidden_size,
            "vocab_size": args.vocab_size,
            "padding_idx": args.pad_token_id,
            "max_position_embeddings": args.max_position_embeddings,
            "scale_embedding": args.scale_embedding,
            "decoder_layers": args.decoder_layers,
            "decoder_attention_heads": args.decoder_attention_heads,
            "activation_function": args.activation_function,
            "dropout": args.dropout or 0.1,
            "decoder_layerdrop": args.decoder_layerdrop or 0.1,
            "attention_dropout": args.attention_dropout or 0.1,
            "activation_dropout": args.activation_dropout or 0.1,
            "decoder_ffn_dim": args.decoder_ffn_dim or 3072,
            "init_std": args.init_std or 0.02,
            "embed_tokens": None
        }

        if args.share_embedding:
            self.embedding = self.context_encoder.get_input_embeddings()
            self.target_encoder.set_input_embeddings(self.embedding)
            args_decoder["embed_tokens"] = self.embedding
        
        self.backward_decoder = TransformerDecoder(**args_decoder)
        self.forward_decoder = TransformerDecoder(**args_decoder)

        self.projection = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.ReLU())
        
        self.lm_backward = nn.Linear(args.hidden_size, args.vocab_size)
        self.lm_forward = nn.Linear(args.hidden_size, args.vocab_size)

        self._init_lm_weights(init_std=args.init_std or 0.02)
    

    def _init_lm_weights(self, init_std=0.02):
        self.lm_backward.weight.data.normal_(mean=0.0, std=init_std)
        self.lm_forward.weight.data.normal_(mean=0.0, std=init_std)
        if self.lm_backward.bias is not None:
            self.lm_backward.bias.data.zero_()
        if self.lm_forward.bias is not None:
            self.lm_forward.bias.data.zero_()
    
    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden
    
    def compute_bi_distance(self, back_hidden_states, for_hidden_states, back_gold_masks, for_gold_masks):
        proj_back = self.projection(back_hidden_states)
        proj_for = self.projection(for_hidden_states)
        avg_dec_back = self.avg_pool(proj_back, mask=back_gold_masks)  # (bsz, hidden_size)
        avg_dec_for = self.avg_pool(proj_for, mask=for_gold_masks)     # (bsz, hidden_size)
        dist_loss = torch.dist(avg_dec_back, avg_dec_for, p=2)

        return dist_loss
    
    def compute_acc(self, lm_logits, seq_masks, seq_labels):
        pred = torch.softmax(lm_logits, -1)
        _, pred_y = pred.max(-1)
        hit_tokens = (torch.eq(pred_y, seq_labels).float()*seq_masks).sum().item()
        num_tokens = seq_masks.float().sum().item()
        acc = float(hit_tokens) / num_tokens if num_tokens > 0 else 0.0
        
        return acc

    def encode(self,
        ctx_input_ids,
        tgt_input_ids, 
        ctx_masks=None,
        tgt_masks=None,
        ctx_seg_ids=None,
        tgt_seg_ids=None
    ):
        """encode context input and target input"""
        ctx_hidden_output = self.context_encoder(input_ids=ctx_input_ids, 
            attention_mask=ctx_masks, token_type_ids=ctx_seg_ids
        )
        ctx_hidden_states = ctx_hidden_output[0]

        tgt_hidden_output = self.target_encoder(input_ids=tgt_input_ids,
            attention_mask=tgt_masks, token_type_ids=tgt_seg_ids
        )
        tgt_hidden_states = tgt_hidden_output[0]
        
        #TODO: concat or cross attn?
        enc_hidden_output = torch.cat([ctx_hidden_states, tgt_hidden_states], dim=1)
        if ctx_masks is not None and tgt_masks is not None:
            enc_attn_mask = torch.cat([ctx_masks, tgt_masks], dim=1)
        else:
            enc_attn_mask = None

        return (enc_hidden_output, enc_attn_mask, tgt_hidden_states, tgt_masks)
    
    def decode(self,
        back_input_ids,
        for_input_ids,
        back_attention_mask=None,
        for_attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        """decode with forward path and backward path"""
        back_output = self.backward_decoder(
            input_ids=back_input_ids,
            attention_mask=back_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False
        )
        back_seq_output = back_output[0]

        for_output = self.forward_decoder(
            input_ids=for_input_ids,
            attention_mask=for_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False
        )
        for_seq_output = for_output[0]

        return (back_seq_output, for_seq_output)

    def forward(self, batch, is_test=False):
        """model training"""
        ctx_input_ids, ctx_masks, ctx_seg_ids = batch["context"]
        tgt_input_ids, tgt_masks, tgt_seg_ids = batch["target"]
        back_plan_ids, back_plan_masks, back_plan_tgt_masks = batch["backward_plan"]
        for_plan_ids, for_plan_masks, for_plan_tgt_masks = batch["forward_plan"]
        
        if is_test:
            back_input_ids = back_plan_ids
            back_attn_masks = back_plan_masks
            for_input_ids = for_plan_ids
            for_attn_masks = for_plan_masks
        else:
            # backward paths
            back_input_ids = back_plan_ids[:, :-1]
            back_attn_masks = back_plan_masks[:, :-1]
            back_plan_gold = back_plan_ids[:, 1:]
            back_plan_gold = back_plan_gold.masked_fill(back_plan_gold == self.pad_token_id, -100)
            back_gold_masks = back_plan_masks[:, 1:]
            back_tgt_masks = back_plan_tgt_masks[:, 1:]
            # negative backward paths
            back_neg_ids, back_neg_attn_masks, back_neg_tgt_masks = batch["backward_neg_plan"]
            back_neg_input = back_neg_ids[:, :, :-1]
            back_neg_masks = back_neg_attn_masks[:, :, :-1]
            back_neg_tgt_masks = back_neg_tgt_masks[:, :, 1:]

            # forward paths
            for_input_ids = for_plan_ids[:, :-1]
            for_attn_masks = for_plan_masks[:, :-1]
            for_plan_gold = for_plan_ids[:, 1:]
            for_plan_gold = for_plan_gold.masked_fill(for_plan_gold == self.pad_token_id, -100)  
            for_gold_masks = for_plan_masks[:, 1:]
            for_tgt_masks = for_plan_tgt_masks[:, 1:]
            # negative forward paths
            for_neg_ids, for_neg_attn_masks, for_neg_tgt_masks = batch["forward_neg_plan"]
            for_neg_input = for_neg_ids[:, :, :-1]
            for_neg_masks = for_neg_attn_masks[:, :, :-1]
            for_neg_tgt_masks = for_neg_tgt_masks[:, :, 1:]
            
            
        enc_hidden_output, enc_attn_mask, tgt_hidden_states, tgt_masks = self.encode(
            ctx_input_ids=ctx_input_ids, tgt_input_ids=tgt_input_ids,
            ctx_masks=ctx_masks, tgt_masks=tgt_masks,
            ctx_seg_ids=ctx_seg_ids, tgt_seg_ids=tgt_seg_ids
        )
        
        back_seq_output, for_seq_output = self.decode(
            back_input_ids=back_input_ids,
            for_input_ids=for_input_ids,
            back_attention_mask=back_attn_masks,
            for_attention_mask=for_attn_masks,
            encoder_hidden_states=enc_hidden_output,
            encoder_attention_mask=enc_attn_mask
        )
        lm_backward_logits = self.lm_backward(back_seq_output)
        lm_forward_logits = self.lm_forward(for_seq_output)
        
        if is_test:
            output = {
                "lm_backward_logits": lm_backward_logits,
                "lm_forward_logits": lm_forward_logits,
                "backward_hidden_states": back_seq_output,
                "forward_hidden_states": for_seq_output
            }
        else:
            ce_criterion = nn.CrossEntropyLoss()
            
            lm_backward_loss = ce_criterion(lm_backward_logits.view(-1, self.vocab_size), back_plan_gold.view(-1))
            lm_forward_loss = ce_criterion(lm_forward_logits.view(-1, self.vocab_size), for_plan_gold.view(-1))
            lm_loss = lm_backward_loss + lm_forward_loss
        
            # min distance
            dist_loss = self.compute_bi_distance(back_seq_output, for_seq_output, back_gold_masks, for_gold_masks)
            
            # enc-dec contrast
            cos_sim = nn.CosineSimilarity(dim=-1)

            proj_enc_tgt = self.projection(tgt_hidden_states)
            avg_enc_target = self.avg_pool(proj_enc_tgt, mask=tgt_masks)   # (bsz, hidden_size)

            proj_back = self.projection(back_seq_output)
            proj_for = self.projection(for_seq_output)
            avg_dec_back_tgt = self.avg_pool(proj_back, mask=back_tgt_masks)  # (bsz, hidden_size)
            avg_dec_for_tgt = self.avg_pool(proj_for, mask=for_tgt_masks)     # (bsz, hidden_size)

            bsz, num_neg, neg_back_len = back_neg_input.size()
            back_neg_input = back_neg_input.reshape(-1, neg_back_len)     # (bsz x num, back_seq_len)
            back_neg_masks = back_neg_masks.reshape(-1, neg_back_len)     # (bsz x num, back_seq_len)
            back_neg_tgt_masks = back_neg_tgt_masks.reshape(-1, neg_back_len)  # (bsz x num, back_seq_len)
            _, _, neg_for_len = for_neg_input.size()
            for_neg_input = for_neg_input.reshape(-1, neg_for_len)   # (bsz x num, for_seq_len)
            for_neg_masks = for_neg_masks.reshape(-1, neg_for_len)   # (bsz x num, for_seq_len)
            for_neg_tgt_masks = for_neg_tgt_masks.reshape(-1, neg_for_len)  # (bsz x num, for_seq_len)

            el_enc_hidden_output = enc_hidden_output.unsqueeze(1).repeat(1, num_neg, 1, 1)
            el_enc_hidden_output = el_enc_hidden_output.reshape(bsz*num_neg, -1, self.hidden_size)
            el_enc_attn_mask = enc_attn_mask.unsqueeze(1).repeat(1, num_neg, 1)
            el_enc_attn_mask = el_enc_attn_mask.reshape(bsz*num_neg, -1)

            neg_back_seq_output, neg_for_seq_output = self.decode(
                back_input_ids=back_neg_input,
                for_input_ids=for_neg_input,
                back_attention_mask=back_neg_masks,
                for_attention_mask=for_neg_masks,
                encoder_hidden_states=el_enc_hidden_output,
                encoder_attention_mask=el_enc_attn_mask
            )

            proj_neg_back = self.projection(neg_back_seq_output)
            proj_neg_for = self.projection(neg_for_seq_output)
            avg_dec_neg_back = self.avg_pool(proj_neg_back, back_neg_tgt_masks)  # (bsz x num, hidden_size)
            avg_dec_neg_for = self.avg_pool(proj_neg_for, for_neg_tgt_masks)     # (bsz x num, hidden_size)
            

            back_sim = cos_sim(avg_enc_target, avg_dec_back_tgt).unsqueeze(1)   # (bsz, 1)
            back_adv_sim = cos_sim(avg_enc_target.unsqueeze(1), avg_dec_neg_back.reshape(bsz, -1, self.hidden_size))  # (bsz, num)
            back_logits = torch.cat([back_sim, back_adv_sim], dim=1) / self.tau      # (bsz, 1 + num)
            
            for_sim = cos_sim(avg_enc_target, avg_dec_for_tgt).unsqueeze(1)
            for_adv_sim = cos_sim(avg_enc_target.unsqueeze(1), avg_dec_neg_for.reshape(bsz, -1, self.hidden_size))
            for_logits = torch.cat([for_sim, for_adv_sim], dim=1) / self.tau

            cont_labels = torch.zeros(bsz, dtype=torch.long, device=ctx_input_ids.device)
            cont_back_loss = ce_criterion(back_logits, cont_labels)
            cont_for_loss = ce_criterion(for_logits, cont_labels)
            cont_loss = 0.5 * (cont_back_loss + cont_for_loss)
            
            # final loss
            loss = lm_loss + self.beta * dist_loss + self.gamma * cont_loss

            backward_acc = self.compute_acc(lm_backward_logits, back_gold_masks, back_plan_gold)
            forward_acc = self.compute_acc(lm_forward_logits, for_gold_masks, for_plan_gold)
            acc = 0.5 * (backward_acc + forward_acc)
            
            output = {
                "loss": loss,
                "acc": acc,
                "lm_backward_loss": lm_backward_loss,
                "lm_forward_loss": lm_forward_loss,
                "dist_loss": dist_loss,
                "cont_loss": cont_loss
            }
        return output

    def generate(self, args, inputs):
        """model inference"""
        # parse args and set accordingly if need
        max_dec_len = args.max_dec_len
        beam_size = args.beam_size
        assert max_dec_len > 0
        assert beam_size >= 1
        
        lambda_scale = args.lambda_scale or 0.1
        min_length = args.min_length or 1
        diversity_penalty = args.diversity_penalty or None
        repetition_penalty = args.repetition_penalty or None
        no_repeat_ngram_size = args.no_repeat_ngram_size or None

        best_back_seq = biconstrained_beam_search(
            model=self,
            inputs=inputs,
            num_beams=beam_size,
            lambda_scale=lambda_scale,
            diversity_penalty=diversity_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_length=min_length,
            max_length=max_dec_len,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id
        )
        
        output_dict = {
            "output_plan": best_back_seq
        }

        return output_dict
