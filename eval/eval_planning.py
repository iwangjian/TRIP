#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
from sklearn import metrics

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

def load_eval_data(fp, lower_case=True):
    actions = []
    topics = []
    with open(fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            act = sample["action"]
            topic = sample["topic"]
            if lower_case:
                act = act.lower()
                topic = topic.lower()
            actions.append(act)
            topics.append(topic)
    assert len(actions) == len(topics)

    action_labels = load_labels(LABEL_ACTION_PATH, lower_case=lower_case)
    topic_labels = load_labels(LABEL_TOPIC_PATH, lower_case=lower_case)
    action_ids = [action_labels.get(act, -1) for act in actions]
    topic_ids = [topic_labels.get(top, -1) for top in topics]
    
    return (action_ids, topic_ids)


def load_gold_data(fp, lower_case=True):
    ids = []
    actions = []
    topics = []
    with open(fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            ids.append(int(sample["id"]))
            act = sample["action_path"][0]       # current action
            topic = sample["topic_path"][0]      # current topic
            if lower_case:
                act = act.lower()
                topic = topic.lower()
            actions.append(act)
            topics.append(topic)
    assert len(ids) == len(actions) == len(topics)

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
    
    return (action_ids, topic_ids, bi_action_ids, bi_topic_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--gold_file", type=str)
    args = parser.parse_args()

    pred_actions, pred_topics = load_eval_data(args.eval_file)
    gold_actions, gold_topics, gold_bi_actions, gold_bi_topics = load_gold_data(args.gold_file)
    
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
