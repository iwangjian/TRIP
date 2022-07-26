#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from sklearn import metrics

TURN_SEP = "#"    # separator of different turns
LABEL_ACTION_PATH = "data_v2/vocab_action.txt"
LABEL_TOPIC_PATH = "data_v2/vocab_topic.txt"

def load_labels(fp, lower_case=True):
    labels = {}
    with open(fp, 'r', encoding='utf-8') as fr:
        for idx, item in enumerate(fr):
            k = item.strip().lower() if lower_case else item.strip()
            labels[k] = idx
    labels["None"] = -1
    return labels


def calc_accuracy(hyps, refs):
    assert len(hyps) == len(refs)
    acc_list = []
    for hyp, ref in zip(hyps, refs):
        if hyp == ref:
            acc_list.append(1)
        else:
            acc_list.append(0)
    acc = np.mean(acc_list)
    return acc

def calc_bi_accuracy(hyps, refs):
    assert len(hyps) == len(refs)
    acc_list = []
    for hyp, ref in zip(hyps, refs):
        if hyp in ref:
            acc_list.append(1)
        else:
            acc_list.append(0)
    acc = np.mean(acc_list)
    return acc

def calc_f1(hyps, refs):
    """Calculate F1 with micro average"""
    assert len(hyps) == len(refs)
    f1 = metrics.f1_score(y_true=refs, y_pred=hyps, average="micro")
    return f1

def calc_bi_f1(hyps, refs):
    """Calculate Bi-F1 with micro average"""
    assert len(hyps) == len(refs)
    golden_total = 0.0
    pred_total = 0.0
    hit_total = 0.0
    for hyp, ref in zip(hyps, refs):
        if hyp in ref:
            hit_total += 1
        golden_total += 1
        pred_total += 1
    p = hit_total / pred_total if pred_total > 0 else 0
    r = hit_total / golden_total if golden_total > 0 else 0
    bi_f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return bi_f1

def calc_bleu(hyps, refs):
    """ Calculate bleu 1/2 """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method1,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method1,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2

def load_eval_data(fp, lower_case=True):
    actions = []
    topics = []
    attrs = []
    with open(fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            act = sample["action"]
            topic = sample["topic"]
            attr = sample.get("attribute", None)
            if lower_case:
                act = act.lower()
                topic = topic.lower()
                if attr is not None:
                    attr = attr.lower()
            
            actions.append(act)
            topics.append(topic)
            if attr is not None:
                attrs.append([tok for tok in attr])
    assert len(actions) == len(topics)
    if len(attrs) == 0:
        attrs = None
    else:
        assert len(actions) == len(attrs)

    action_labels = load_labels(LABEL_ACTION_PATH, lower_case=lower_case)
    topic_labels = load_labels(LABEL_TOPIC_PATH, lower_case=lower_case)
    action_ids = [action_labels.get(act, -1) for act in actions]
    topic_ids = [topic_labels.get(top, -1) for top in topics]
    
    return (action_ids, topic_ids, attrs)


def load_gold_data(fp, lower_case=True):
    ids = []
    actions = []
    topics = []
    attrs = []
    with open(fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            ids.append(int(sample["id"]))
            
            act = sample["action_path"][0]       # current action
            topic = sample["topic_path"][0]      # current topic
            attr_str = ""
            #cur_stage = sample["plan"]["attribute"][0]   # current stage's attribute
            #for attr in cur_stage:
            #    attr_str += "".join(attr)
            #    attr_str += TURN_SEP
            if lower_case:
                act = act.lower()
                topic = topic.lower()
                attr_str = attr_str.lower()
            actions.append(act)
            topics.append(topic)
            attrs.append([tok for tok in attr_str])
    assert len(ids) == len(actions) == len(topics) == len(attrs)

    action_labels = load_labels(LABEL_ACTION_PATH, lower_case=lower_case)
    topic_labels = load_labels(LABEL_TOPIC_PATH, lower_case=lower_case)
    action_ids = [action_labels[act] for act in actions]
    topic_ids = [topic_labels[top] for top in topics]
        
    bi_action_ids = []
    bi_topic_ids = []
    prev_id = -1
    for idx, cur_id in enumerate(ids):
        if cur_id == prev_id:
            bi_acts = [action_ids[idx-1], action_ids[idx]]
            bi_tops = [topic_ids[idx-1], topic_ids[idx]]
        else:
            bi_acts = [action_ids[idx]]
            bi_tops = [topic_ids[idx]]
        bi_action_ids.append(bi_acts)
        bi_topic_ids.append(bi_tops)
        prev_id = cur_id
    
    return (action_ids, topic_ids, bi_action_ids, bi_topic_ids, attrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--gold_file", type=str)
    args = parser.parse_args()

    pred_actions, pred_topics, pred_attrs = load_eval_data(args.eval_file)
    gold_actions, gold_topics, gold_bi_actions, gold_bi_topics, gold_attrs = load_gold_data(args.gold_file)
    
    # calculate f1
    action_f1 = calc_f1(pred_actions, gold_actions)
    topic_f1 = calc_f1(pred_topics, gold_topics)

    # calculate bi-f1
    action_bi_f1 = calc_bi_f1(pred_actions, gold_bi_actions)
    topic_bi_f1 = calc_bi_f1(pred_topics, gold_bi_topics)

    # calculate bleu
    bleu_1, bleu_2 = None, None
    if pred_attrs is not None:
        bleu_1, bleu_2 = calc_bleu(pred_attrs, gold_attrs)

    output_str = "Action F1: %.2f%%\n" % (action_f1 * 100)
    output_str += "Action Bi-F1: %.2f%%\n" % (action_bi_f1 * 100)
    output_str += "Topic F1: %.2f%%\n" % (topic_f1 * 100)
    output_str += "Topic Bi-F1: %.2f%%" % (topic_bi_f1 * 100)
    if bleu_1 is not None and bleu_2 is not None:
        output_str += "\nAttribute BLEU1: %.3f\n" % bleu_1
        output_str += "Attribute BLEU2: %.3f" % bleu_2

    print(output_str)
