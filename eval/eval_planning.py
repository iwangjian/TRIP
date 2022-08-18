#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import json
from sklearn import metrics


def load_labels(fp, lower_case=True):
    labels = {}
    with open(fp, 'r', encoding='utf-8') as fr:
        for idx, item in enumerate(fr):
            k = "".join(item.strip().split())   # concat all chars
            if lower_case:
                k = k.lower()
            labels[k] = idx
    labels["None"] = -1
    return labels

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

def load_eval_data(eval_fp, label_action_fp, label_topic_fp, lower_case=True):
    actions = []
    topics = []
    with open(eval_fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            act = "".join(sample["action"].strip().split())   # concat all action chars
            topic = "".join(sample["topic"].strip().split())  # concat all topic chars
            if lower_case:
                act = act.lower()
                topic = topic.lower()
            actions.append(act)
            topics.append(topic)
    assert len(actions) == len(topics)

    action_labels = load_labels(label_action_fp, lower_case=lower_case)
    topic_labels = load_labels(label_topic_fp, lower_case=lower_case)
    action_ids = [action_labels.get(act, -1) for act in actions]
    topic_ids = [topic_labels.get(top, -1) for top in topics]
    
    return (action_ids, topic_ids)


def load_gold_data(gold_fp, label_action_fp, label_topic_fp, lower_case=True):
    ids = []
    actions = []
    topics = []
    with open(gold_fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            ids.append(int(sample["id"]))
            act = "".join(sample["action_path"][0].strip().split())   # concat all action chars
            topic = "".join(sample["topic_path"][0].strip().split())  # concat all topic chars
            if lower_case:
                act = act.lower()
                topic = topic.lower()
            actions.append(act)
            topics.append(topic)
    assert len(ids) == len(actions) == len(topics)

    action_labels = load_labels(label_action_fp, lower_case=lower_case)
    topic_labels = load_labels(label_topic_fp, lower_case=lower_case)
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
    
    return (action_ids, topic_ids, bi_action_ids, bi_topic_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--gold_file", type=str)
    args = parser.parse_args()

    label_dir = os.path.dirname(args.gold_file)
    label_action_fp = os.path.join(label_dir, "vocab_action.txt")
    label_topic_fp = os.path.join(label_dir, "vocab_topic.txt")
    if not os.path.exists(label_action_fp):
        raise FileExistsError("{} not exit!".format(label_action_fp))
    if not os.path.exists(label_topic_fp):
        raise FileExistsError("{} not exit!".format(label_topic_fp))

    pred_actions, pred_topics = load_eval_data(args.eval_file, label_action_fp, label_topic_fp)
    gold_actions, gold_topics, gold_bi_actions, gold_bi_topics = load_gold_data(args.gold_file, label_action_fp, label_topic_fp)
    
    # calculate f1
    action_f1 = calc_f1(pred_actions, gold_actions)
    topic_f1 = calc_f1(pred_topics, gold_topics)

    # calculate bi-f1
    action_bi_f1 = calc_bi_f1(pred_actions, gold_bi_actions)
    topic_bi_f1 = calc_bi_f1(pred_topics, gold_bi_topics)

    output_str = "Action F1: %.2f%%\n" % (action_f1 * 100)
    output_str += "Action Bi-F1: %.2f%%\n" % (action_bi_f1 * 100)
    output_str += "Topic F1: %.2f%%\n" % (topic_f1 * 100)
    output_str += "Topic Bi-F1: %.2f%%" % (topic_bi_f1 * 100)

    print(output_str)
