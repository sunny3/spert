import json
import argparse
import os

if os.getcwd().find('converters')>=0:
    os.chdir("../")

parser = argparse.ArgumentParser(description='Help to parse args')
parser.add_argument('--data_path', type=str, help='Path to the dataset, whith test.json and predictions_valid_epoch_<epoch_num>.json files')

args = parser.parse_args()

ds_path = args.data_path

test_file = os.path.join(ds_path, 'test_spert.json')
pred_files = [os.path.join(ds_path, file) for file in os.listdir(ds_path) if file.find('predictions_valid_epoch_')>=0]
if len(pred_files)!=1:
    raise ValueError("Error, dataset directory must contain only one file which include the substring 'predictions_valid_epoch_'")

pred_file = pred_files[0]

with open(pred_file, 'r') as f:
    pred_data = json.load(f)
with open(test_file, 'r') as f:
    test_data_with_spans = json.load(f)
    
assert len(test_data_with_spans) == len(test_data_with_spans)


for pred_rev, test_rev in zip(pred_data, test_data_with_spans):
    assert len(test_rev['tok_spans']) == len(pred_rev['tokens'])
    test_rev['entities'] = {}
    for ent_num, pred_ent in enumerate(pred_rev['entities']):
        test_rev['entities'][ent_num] = dict.fromkeys(['spans', 'tag'], None)

        begin = test_rev['tok_spans'][int(pred_ent['start'])][0]
        end = test_rev['tok_spans'][int(pred_ent['end'])-1][1]

        test_rev['entities'][ent_num]['spans'] = [{'begin': begin, 'end': end}]
        test_rev['entities'][ent_num]['tag'] = [pred_ent['type']]
        test_rev['entities'][ent_num]['text'] = test_rev['text'][begin:end]
    test_rev['relations'] = []
    for pred_rel in pred_rev['relations']:
        rel_type = pred_rel['type'] #if pred_rel['type'][-1]=='1' else 'none_' + pred_rel['type']
        test_rev['relations'].append({'first_entity': test_rev['entities'][int(pred_rel['head'])],
                                      'second_entity': test_rev['entities'][int(pred_rel['tail'])],
                                      'relation_type': rel_type})
with open(ds_path + '/pred.json', 'w') as f:
    json.dump(test_data_with_spans, f)
print('Spert predictions saved in pred.py, path: %s'%ds_path)