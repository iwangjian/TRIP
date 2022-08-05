# -*- coding: utf-8 -*-
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from transformers import GPT2LMHeadModel
from model.nn_decoder import TransformerDecoder

SMALL_CONST = 1e-15


class TRIPDialGPT2(nn.Module):
    """
    Model class: TRIPDialGPT2
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
        self.use_control = args.use_control
        
        self.lm_decoder = GPT2LMHeadModel.from_pretrained(args.config_dir)
        self.lm_decoder.resize_token_embeddings(args.vocab_size)
        
        # whether to use a control model p(a|y) to control generation
        if self.use_control:
            args_controller = {
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
            self.controller = TransformerDecoder(**args_controller)
            if args.share_embedding:
                # share?
                #embeds = self.lm_decoder.get_input_embeddings()
                #self.controller.set_input_embeddings(embeds)
                # copy?
                embeds = self.lm_decoder.get_input_embeddings()
                self.controller.embed_tokens.weight.data.copy_(embeds.weight.clone().detach())

            self.lm_control = nn.Linear(args.hidden_size, args.vocab_size)
            self._init_lm_weights(init_std=args.init_std or 0.02)
    

    def _init_lm_weights(self, init_std=0.02):
        self.lm_control.weight.data.normal_(mean=0.0, std=init_std)
        if self.lm_control.bias is not None:
            self.lm_control.bias.data.zero_()
    
    def compute_acc(self, lm_logits, seq_masks, seq_labels):
        pred = torch.softmax(lm_logits, -1)
        _, pred_y = pred.max(-1)
        hit_tokens = (torch.eq(pred_y, seq_labels)*seq_masks).sum().item()
        num_tokens = seq_masks.sum().item()
        acc = float(hit_tokens) / num_tokens if num_tokens > 0 else 0.0
        
        return acc
    
    @staticmethod
    def _to_var(x, requires_grad=False, volatile=False, device='cuda'):
        x = x.to(device)
        return Variable(x, requires_grad=requires_grad, volatile=volatile)
    
    @staticmethod
    def _add_func(L1, L2):
        return [[m1 + m2 for (m1, m2) in zip(l1, l2)]
                for (l1, l2) in zip(L1, L2)
        ]
    
    def perturb_past(self, 
        past, 
        last,
        ctrl_ids,
        ctrl_mask,
        unpert_hidden,
        unpert_logits,
        stepsize=0.03,
        num_iterations=3,
        gamma=1.5,
        bow_scale=0.01,
        ce_scale=1,
        kl_scale=0.01
        ):
        """perturb past to control a LM"""
        last = last.detach()
        device = last.device
        ctrl_ids = ctrl_ids.detach()
        ctrl_mask = ctrl_mask.detach()
        unpert_hidden = unpert_hidden.detach()
        unpert_hidden.requires_grad = True
        unpert_logits = unpert_logits.detach()
        unpert_logits.requires_grad = True
        past = [[p.detach() for p in p_layer]
            for p_layer in past
        ]
        past = [[self._to_var(p, requires_grad=True, device=device) for p in p_layer]
            for p_layer in past
        ]
        
        ctrl_input_ids = ctrl_ids[:, :-1]
        ctrl_attn_masks = ctrl_mask[:, :-1] if ctrl_mask is not None else None
        ctrl_gold_ids = ctrl_ids[:, 1:]
        ctrl_gold_ids = ctrl_gold_ids.masked_fill(
            ctrl_gold_ids == self.pad_token_id, -100  # default: ignore index -100 
        )   

        # init perturbed past
        grad_accumulator = [[(torch.zeros(p.shape, dtype=p.dtype).to(device)) for p in p_layer]
            for p_layer in past
        ]
        
        # accumulate perturbations for num_iterations
        for i in range(num_iterations):
            curr_perturbation = [[self._to_var(p_, requires_grad=True, device=device) for p_ in p_layer]
                for p_layer in grad_accumulator
            ]

            # compute hidden using perturbed past
            perturbed_past = self._add_func(past, curr_perturbation)
        
            pert_lm_output = self.lm_decoder(
                input_ids=last,
                past_key_values=perturbed_past,
                output_hidden_states=True,
                return_dict=True
            )
            pert_logits = pert_lm_output["logits"]
            pert_hidden = pert_lm_output["hidden_states"][-1]

            loss = 0.0
            
            # computre bow loss
            log_probs = F.log_softmax(pert_logits, dim=-1)
            one_hot_bows = F.one_hot(ctrl_ids[:, 1:-1], self.vocab_size)
            bow_loss = -1 * torch.sum(one_hot_bows * log_probs, 1)
            bow_loss = bow_loss.sum()
            loss += bow_scale * bow_loss
            '''
            # compute controller loss
            con_hidden = torch.cat([unpert_hidden[:, :-1, :], pert_hidden], dim=1).detach()

            ctrl_output = self.controller(
                input_ids=ctrl_input_ids,
                attention_mask=ctrl_attn_masks,
                encoder_hidden_states=con_hidden,
                use_cache=False
            )
            ctrl_dec_hidden = ctrl_output[0]
            ctrl_logits = self.lm_control(ctrl_dec_hidden)
            #ctrl_logits = ctrl_logits.detach()
            #ctrl_logits.requires_grad = True
            ce_criterion = nn.CrossEntropyLoss()
            ce_loss = ce_criterion(ctrl_logits.view(-1, self.vocab_size), ctrl_gold_ids.view(-1))
            loss += ce_scale * ce_loss
            '''

            if kl_scale > 0:
                unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
                unpert_probs = unpert_probs + SMALL_CONST * (
                    unpert_probs <= SMALL_CONST).float().to(device)
                #unpert_probs = unpert_probs.detach()
                #unpert_probs.requires_grad = True
                
                pert_probs = F.softmax(pert_logits[:, -1, :], dim=-1)
                correction = SMALL_CONST * (pert_probs <= SMALL_CONST).float().to(device)
                corrected_probs = pert_probs + correction
                #corrected_probs = corrected_probs.detach()
                #corrected_probs.requires_grad = True
    
                kl_loss = (corrected_probs * (corrected_probs / unpert_probs).log()).sum()        
                loss += kl_scale  * kl_loss

            # compute gradients
            #print("iter: %d loss: %.3f bow_loss: %.3f ce_loss: %.3f kl_loss: %.3f" % (i, loss, bow_loss, ce_loss, kl_loss))
            loss.backward()

            grad_norms = [[torch.norm(p_.grad) + SMALL_CONST for p_ in p_layer]
                for p_layer in curr_perturbation
            ]
            grad = [[-stepsize * (p_.grad / grad ** gamma).data for grad, p_ in zip(grads, p_layer)] 
                for grads, p_layer in zip(grad_norms, curr_perturbation)
            ]

            # accumulate gradients
            grad_accumulator = self._add_func(grad, grad_accumulator)

            # reset gradients
            for p_layer in curr_perturbation:
                for p_ in p_layer:
                    p_.grad.data.zero_()

            # removing past from the graph
            new_past = []
            for p_layer in past:
                new_past.append([])
                for p_ in p_layer:
                    new_past[-1].append(p_.detach())
            past = new_past
            
        
        # apply the accumulated perturbations to the past
        grad_accumulator = [[self._to_var(p_, requires_grad=True, device=device).detach() for p_ in p_layer]
            for p_layer in grad_accumulator
        ]
        pert_past = self._add_func(past, grad_accumulator)

        return pert_past


    def forward(self, batch, is_test=False):
        if is_test:
            input_ids = batch["input_ids"]
            
            # run LM model forward to obtain unperturbed hidden output
            lm_output = self.lm_decoder(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            lm_hidden = lm_output["hidden_states"][-1]
            lm_logits = lm_output["logits"]
            
            if self.use_control:
                control_ids  = batch["control_ids"]
                control_masks = batch["control_masks"]
                past = lm_output["past_key_values"]
                
                past_l = []
                for p_layer in past:
                    p_0 = p_layer[0][:, :, :-1, :]
                    p_1 = p_layer[1][:, :, :-1, :]
                    past_l.append([p_0, p_1])
                last_l = input_ids[:, -1:]
                
                # obtain perturbed hidden output
                perturbed_past = self.perturb_past(
                    past=past_l,
                    last=last_l,
                    ctrl_ids=control_ids,
                    ctrl_mask=control_masks,
                    unpert_hidden=lm_hidden,
                    unpert_logits=lm_logits,
                )
                # run LM model forward with perturbed past key-values
                pert_lm_output = self.lm_decoder(
                    input_ids=last_l,
                    past_key_values=perturbed_past,
                    return_dict=True
                )
                lm_logits = pert_lm_output["logits"]
            
            output = {
                "logits": lm_logits,
            }
        else:
            input_ids = batch["input_ids"]
            lm_labels = batch["lm_labels"]
            lm_labels = lm_labels.masked_fill(lm_labels == self.pad_token_id, -100)   # default: ignore index -100
            label_masks = lm_labels.masked_fill(lm_labels == -100, 0)
            label_masks = label_masks.masked_fill(label_masks > 0, 1)

            # run LM model
            lm_output = self.lm_decoder(
                input_ids=input_ids,
                labels=lm_labels,
                output_hidden_states=True,
                return_dict=True
            )
            lm_hidden = lm_output["hidden_states"][-1]
            lm_logits = lm_output["logits"]
            lm_loss = lm_output["loss"]
            
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            shift_masks = label_masks[:, 1:].contiguous()
            acc = self.compute_acc(shift_logits, shift_masks, shift_labels)
            loss = lm_loss

            if self.use_control:
                control_ids  = batch["control_ids"]
                control_masks = batch["control_masks"]
                ctrl_input_ids = control_ids[:, :-1]
                ctrl_attn_masks = control_masks[:, :-1] if control_masks is not None else None
                ctrl_gold_ids = control_ids[:, 1:]
                ctrl_gold_ids = ctrl_gold_ids.masked_fill(ctrl_gold_ids == self.pad_token_id, -100)    # default: ignore index -100

                # run controller
                ctrl_output = self.controller(
                    input_ids=ctrl_input_ids,
                    attention_mask=ctrl_attn_masks,
                    encoder_hidden_states=lm_hidden,
                    use_cache=False
                )
                ctrl_dec_hidden = ctrl_output[0]
                ctrl_logits = self.lm_control(ctrl_dec_hidden)

                ce_criterion = nn.CrossEntropyLoss()
                lm_ctrl_loss = ce_criterion(ctrl_logits.view(-1, self.vocab_size), ctrl_gold_ids.view(-1))
                loss += 0.1 * lm_ctrl_loss

                output = {
                    "loss": loss,
                    "lm_loss": lm_loss,
                    "lm_ctrl_loss": lm_ctrl_loss,
                    "acc": acc
                }
            else:
                output = {
                    "loss": loss,
                    "lm_loss": lm_loss,
                    "acc": acc
                }
        
        return output


    def generate(self, args, inputs):
        """model inference"""
        n_ctx = self.lm_decoder.config.n_ctx
        special_tokens_ids = [self.pad_token_id, self.eos_token_id]
        max_dec_len = args.max_dec_len
        assert max_dec_len > 0

        input_ids = inputs["input_ids"]
        batch_size = input_ids.size(0)
        output_ids = input_ids.new(batch_size, max_dec_len).fill_(self.pad_token_id)

        for batch_idx in range(batch_size):
            idx_inputs = {
                "input_ids": inputs["input_ids"][batch_idx:batch_idx+1],
                "control_ids": inputs["control_ids"][batch_idx:batch_idx+1],
                "control_masks": inputs["control_masks"][batch_idx:batch_idx+1]
            }
            cur_input_ids = idx_inputs["input_ids"]
            for len_idx in range(max_dec_len):
                cur_input_ids = cur_input_ids[:, -(n_ctx - 1):]  # (1, seq_len)
                idx_inputs["input_ids"] = cur_input_ids

                lm_output = self.forward(idx_inputs, is_test=True)
                
                logits = lm_output["logits"]
                logits = logits[0, -1, :] / args.temperature
                if args.top_k > 0 or (args.top_p > 0 and args.top_p <= 1):
                    filtered_logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.topk(probs, 1)[1]
                
                if len_idx < args.min_dec_len and next_token.item() in special_tokens_ids:
                    while next_token.item() in special_tokens_ids:
                        next_token = torch.multinomial(probs, num_samples=1)

                output_ids[batch_idx, len_idx] = next_token
                cur_input_ids = torch.cat([cur_input_ids, next_token.unsqueeze(0)], dim=1)

                if next_token.item() in special_tokens_ids:
                    break

        output_dict = {
            "response": output_ids
        }

        return output_dict


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits