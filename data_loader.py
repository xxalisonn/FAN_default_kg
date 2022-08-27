import json
import random
import logging

def train_generate(dataset, batch_size, few, symbol2id, ent2id, id2ent, rel2id, e1rel_e2,dom_ent,ent_dom,rel2dom_h,rel2nn):
    logging.info('Loading Train Data')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('Loading Candidates')
    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    # 取出所有存在的三元组关系
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)

    # 生成fewnum对应的support集
    rel_idx = 0
    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)
        # 选中一个关系任务
        task_choice = task_pool[rel_idx % num_tasks]
        rel_idx += 1
        candidates = rel2candidates[task_choice]
        if len(candidates) <= 20:
            continue
        task_triples = train_tasks[task_choice]
        random.shuffle(task_triples)

        support_triples = task_triples[:few]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
        support_left = [ent2id[triple[0]] for triple in support_triples]
        support_right = [ent2id[triple[2]] for triple in support_triples]

        # 选batchsize个三元组作为训练的query集
        other_triples = task_triples[few:]
        if len(other_triples) == 0:
            continue
        if len(other_triples) < batch_size:
            query_triples = [random.choice(other_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(other_triples, batch_size)

        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]
        query_left = [ent2id[triple[0]] for triple in query_triples]
        query_right = [ent2id[triple[2]] for triple in query_triples]

        # 负样本生成
        false_pairs = []
        false_left = []
        false_right = []
        for triple in query_triples:
            e_h = triple[0]
            rel = triple[1]
            e_t = triple[2]
            neg_candidates = concept_filter_t(e_t, rel, rel2id, rel2dom_h, rel2nn,ent_dom,dom_ent)
            while True:
                # noise = random.choice(candidates)  # select noise from candidates
                # if (noise not in e1rel_e2[e_h + rel]) \
                #         and noise != e_t:
                #     break
                noise = random.choice(neg_candidates)
                noise = id2ent[noise]
                if (noise not in e1rel_e2[e_h + rel]) and noise != e_t:
                    break

            false_pairs.append([symbol2id[e_h], symbol2id[noise]])
            false_left.append(ent2id[e_h])
            false_right.append(ent2id[noise])

        yield support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right

def concept_filter_t(tail, relation, rel2id, rel_t, rel2nn, ent_conc, conc_ents):
    relation = rel2id[relation]
    if str(relation) not in rel_t:
        return []
    rel_tc = rel_t[str(relation)]
    set_tc = set(rel_tc)
    t = []
    if rel2nn[str(relation)] == 2 or rel2nn[str(relation)] == 3:
        if tail in ent_conc:
            for conc in ent_conc[str(tail)]:
                for ent in conc_ents[str(conc)]:
                    t.append(ent)
        else:
            for tc in rel_tc:
                for ent in conc_ents[str(tc)]:
                    t.append(ent)
    else:
        if str(tail) in ent_conc:
            set_ent_conc = set(ent_conc[str(tail)])
        else:
            set_ent_conc = set([])
        set_diff = set_tc - set_ent_conc
        set_diff = list(set_diff)
        for conc in set_diff:
            for ent in conc_ents[str(conc)]:
                t.append(ent)
    t = set(t)
    neg = list(t)

    return neg