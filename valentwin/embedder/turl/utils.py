import copy
import os

RESERVED_ENT_VOCAB = {0:{'wiki_id':'[PAD]', 'wiki_title':'[PAD]', 'count': -1, 'mid': -1},
                        1:{'wiki_id':'[ENT_MASK]','wiki_title':'[ENT_MASK]', 'count': -1, 'mid': -1},
                        2:{'wiki_id':'[PG_ENT_MASK]','wiki_title':'[PG_ENT_MASK]', 'count': -1, 'mid': -1},
                        3:{'wiki_id':'[CORE_ENT_MASK]','wiki_title':'[CORE_ENT_MASK]', 'count': -1, 'mid': -1}
                        }
RESERVED_ENT_VOCAB_NUM = len(RESERVED_ENT_VOCAB)

def load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=1, title_as_key=False):
    entity_vocab = copy.deepcopy(RESERVED_ENT_VOCAB)
    bad_title = 0
    few_entity = 0
    with open(os.path.join(data_dir, 'entity_vocab.txt'), 'r', encoding="utf-8") as f:
        for line in f:
            _, entity_id, entity_title, entity_mid, count = line.strip().split('\t')
            if ignore_bad_title and entity_title == '':
                bad_title += 1
            elif int(count) < min_ent_count:
                few_entity += 1
            else:
                if title_as_key:
                    entity_vocab[entity_title] = {
                        'wiki_id': int(entity_id),
                        'wiki_title': entity_title,
                        'mid': entity_mid,
                        'count': int(count)
                    }
                else:
                    entity_vocab[len(entity_vocab)] = {
                        'wiki_id': int(entity_id),
                        'wiki_title': entity_title,
                        'mid': entity_mid,
                        'count': int(count)
                    }
    print('total number of entity: %d\nremove because of empty title: %d\nremove because count<%d: %d'%(len(entity_vocab),bad_title,min_ent_count,few_entity))
    return entity_vocab