import json
from ds import split_doc_on_words
import os
import argparse

if os.getcwd().find('converters')>=0:
    os.chdir("../")

parser = argparse.ArgumentParser(description='Help to parse args')
parser.add_argument('--data_path', type=str, help='Path to the dataset, which will be converted to the spert format, this path must include files which have test and train substringes in their names')
parser.add_argument('--res_path', type=str, help='This is the relative path from ./data/datasets folder. The full path is ./data/datasets/ + res_dir')

args = parser.parse_args()

ds_path = args.data_path
res_path = args.res_path

if res_path.find('/data/datasets/')>0:
    raise ValueError('Error, --res_path must be relative path from ./data/datasets folder')
    
train_files = [os.path.join(ds_path, file) for file in os.listdir(ds_path) if file.find('train')>=0]
if len(train_files)!=1:
    raise ValueError("Error, dataset directory must contain only one file with the name, which include the substring 'train'")

test_files = [os.path.join(ds_path, file) for file in os.listdir(ds_path) if file.find('test')>=0]
if len(test_files)!=1:
    raise ValueError("Error, dataset directory must contain only one file with the name, which include the substring 'test'")
    

    
train_file = train_files[0]
test_file = test_files[0]

def tokenization(text):
    data_df = split_doc_on_words({'text': text}, language='other')
    toks = []
    for row in data_df.iterrows():
        new_tok = dict.fromkeys(['forma', 'posStart', 'posEnd', 'len'])
        new_tok['posStart'] = row[1]['word_start_in_doc']
        new_tok['posEnd'] = row[1]['word_stop_in_doc']
        new_tok['forma'] = row[1]['word']
        new_tok['len'] = len(new_tok['forma'])
        new_tok['id'] = row[0]
        toks.append(new_tok)
    return toks


with open('./configs/spert_config_example_xlmroberta.conf', 'r') as f:
    spert_config = f.read()

ent_types_set = set()
rel_types_set = set()


print('train file path:\n%s'%(train_file))
print('test file path:\n%s'%(test_file))

for mode, mode_path in [('train', train_file), ('test', test_file)]:
    with open(mode_path) as f:
        data = json.load(f)
    spert_data = []
    for k, uniq_id in enumerate(data):
        #я считаю, что разметка неразрывная, ибо только её и берет spert
        #поэтому от разрывных сущностей я беру только первую часть
        ent_spans = [list(ent['spans'][0].values()) for ent in uniq_id['entities'].values()]
        ent_ids = list(uniq_id['entities'].keys())
        #также берем лишь один тег
        ent_types = [ent['tag'][0] for ent in uniq_id['entities'].values()] 
        text = uniq_id['text']

        new_sample = {}
        toks = tokenization(text)

        new_sample['tokens'] = toks

        #заполняем поле entities в spert формате
        new_sample['entities'] = []
        ent_num = 0
        for ann, ann_id, ann_type in zip(ent_spans, ent_ids, ent_types):
            new_ent = dict.fromkeys(['type', 'start', 'end', 'text', 'origin_entity_id'])
            new_ent['id']=ann_id
            #находим токены
            e_start = ann[0]
            e_end = ann[1]         
            ent_toks = []

            for tok in new_sample['tokens']:
                cond_1 = (tok['posStart'] >= e_start) & (
                        tok['posEnd'] <= e_end)
                cond_4 = (tok['posStart'] <= e_start) & (
                        tok['posEnd'] >= e_end)
                cond_2 = (tok['posStart'] < e_start) & (
                        tok['posStart'] < e_end) & (
                        tok['posEnd'] > e_start)
                cond_3 = (tok['posStart'] > e_start) & (
                        tok['posEnd'] > e_end) & (
                        tok['posStart'] < e_end)

                if (cond_1 | cond_2 | cond_3 | cond_4):
                    ent_toks.append(tok)

            new_ent['type']=ann_type
            try:
                new_ent['start']=ent_toks[0]['id']
                new_ent['end']=ent_toks[-1]['id']+1
            except:
                print('In %s file in review %s entity number %s was lost'%(mode_path, k, ent_num))
                ent_num +=1
                continue
            ent_num +=1
            new_ent['text']=' '.join([tok['forma'] for tok in ent_toks])
            new_sample['entities'].append(new_ent)
            ent_types_set.add(ann_type)
        new_sample['tokens'] = [tok['forma'] for tok in toks]
        new_sample['tok_spans'] = [[tok['posStart'], tok['posEnd']] for tok in toks]
        new_sample['text'] = uniq_id['text']
        #заполняем поле с relations
        new_sample['relations']=[]
        for r_num, rel in enumerate(uniq_id['relations']):
            rel_type = rel['relation_type'].replace('none_', '') + '_' + str(rel['relation_class'])
            rel_types_set.add(rel_type)
            new_rel = dict.fromkeys(['type', 'head', 'tail'])
            new_rel['type'] = rel_type
            for ent_id, ent in enumerate(new_sample['entities']):
                if str(ent['id']) == str(rel['first_entity']['entity_id']):
                    new_rel['head'] = ent_id
                if str(ent['id']) == str(rel['second_entity']['entity_id']):
                    new_rel['tail'] = ent_id
            if new_rel['head'] is None or new_rel['tail'] is None:
                print('%In %s file in review %s relation number %s was lost'%(mode_path, k, r_num))
                continue
            new_sample['relations'].append(new_rel)
        spert_data.append(new_sample)
    save_path = os.path.join('./data/datasets/', res_path) 
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + '/'  + mode + '_spert' + '.json', 'w') as f:
        json.dump(spert_data, f)
    print('%s file in spert format saved in:\n%s'%(mode, os.path.join(save_path, mode + '_spert' + '.json')))

#типы спертовские дампим
types_d = {k: {} for k in ['entities', 'relations']}

for ent_type in ent_types_set:
    types_d['entities'][ent_type] = {'short': ent_type,
                                     'verbose': ent_type}

for rel_type in rel_types_set:
    types_d['relations'][rel_type] = {'short': rel_type,
                                      'verbose': rel_type,
                                      'symmetric': False}

with open(save_path + '/' + 'types' + '.json', 'w') as f:
    json.dump(types_d, f)

#конфиг спертовский дампим
save_path = './configs/'

ds_name = res_path.replace('/', '_')
ds_name = ds_name.strip('_')
ds_name = ds_name.replace('folds', 'fold')

fold_conf = spert_config.replace('data/datasets/RDRS_multicontext/folds_4_classes/1/RDRS_spert_train.json',
                                 os.path.join('data/datasets/', res_path, 'train_spert' + '.json'))
fold_conf = fold_conf.replace('data/datasets/RDRS_multicontext/folds_4_classes/1/RDRS_spert_test.json',
                                 os.path.join('data/datasets/', res_path, 'test_spert' + '.json'))
fold_conf = fold_conf.replace('data/datasets/RDRS_multicontext/folds_4_classes/1/RDRS_types.json',
                                 os.path.join('data/datasets/', res_path, 'types' + '.json'))
fold_conf = fold_conf.replace('RDRS_multicontext_fold_1', ds_name)

print('Config, which will be used for train:\n%s'%(ds_name + '_' + 'train' '.conf'))
print('DS label:\n%s'%ds_name)

with open(save_path + ds_name + '_' + 'train' '.conf', 'w') as f:
    f.write(fold_conf)
