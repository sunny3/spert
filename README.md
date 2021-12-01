# Pipeline SpERT c xlmroberta
Запуск содержится в файле single_spert_exp.sh. Запускать командой `. single_spert_exp.sh`, предварительно выставив желаемые гиперпараметры в файле configs/spert_config_example_xlmroberta.conf и заполнив три переменные в single_spert_exp.sh скрипте:
- DS_path - путь до датасета в нашем формате. Это директория, должна содержать файлы train и test, или если хочется, то немножко переименнованные файлы train и test, главное, чтобы в их названии сохранились подстроки train и test
- RES_path - относительный путь к папке ./data/datasets, в который сохраняется конвертированный в SpERT формат датасет, а также результаты эксперимента (output_log.txt), с файлом выходов модели в SpERT формате (predictions_valid_epoch_<num epoch>.json) и в нашем формате (pred.json)
- MODEL_path - путь к модели. Может быть как веб-путем, так и путем до файлов на диске. Путь к модели должен содержать как файлы самой модели, так и файлы токенизатора этой модели. Достоверно известно, что pipeline может работать с моделями следующих типов: xlm-roberta-base и xlm-roberta-large.
  
При запуске pipeline, если SpERT не предсказал ни одной сущности и отношения, может быть ошибка в скрипте spert_eval.py

# SpERT: Span-based Entity and Relation Transformer
PyTorch code for SpERT: "Span-based Entity and Relation Transformer". For a description of the model and experiments, see our paper: https://arxiv.org/abs/1909.07755 (published at ECAI 2020).

![alt text](http://deepca.cs.hs-rm.de/img/deepca/spert.png)

## Setup
### Requirements
- Required
  - Python 3.5+
  - PyTorch (tested with version 1.4.0)
  - transformers (+sentencepiece, e.g. with 'pip install transformers[sentencepiece]', tested with version 4.1.1)
  - scikit-learn (tested with version 0.24.0)
  - tqdm (tested with version 4.55.1)
  - numpy (tested with version 1.17.4)
- Optional
  - jinja2 (tested with version 2.10.3) - if installed, used to export relation extraction examples
  - tensorboardX (tested with version 1.6) - if installed, used to save training process to tensorboard
  - spacy (tested with version 3.0.1) - if installed, used to tokenize sentences for prediction

### Fetch data
Fetch converted (to specific JSON format) CoNLL04 \[1\] (we use the same split as \[4\]), SciERC \[2\] and ADE \[3\] datasets (see referenced papers for the original datasets):
```
bash ./scripts/fetch_datasets.sh
```

Fetch model checkpoints (best out of 5 runs for each dataset):
```
bash ./scripts/fetch_models.sh
```
The attached ADE model was trained on split "1" ("ade_split_1_train.json" / "ade_split_1_test.json") under "data/datasets/ade".

## Examples
(1) Train CoNLL04 on train dataset, evaluate on dev dataset:
```
python ./spert.py train --config configs/example_train.conf
```

(2) Evaluate the CoNLL04 model on test dataset:
```
python ./spert.py eval --config configs/example_eval.conf
```

(3) Use the CoNLL04 model for prediction. See the file 'data/datasets/conll04/conll04_prediction_example.json' for supported data formats. You have three options to specify the input sentences, choose the one that suits your needs. If the dataset contains raw sentences, 'spacy' must be installed for tokenization. Download a spacy model via 'python -m spacy download model_label' and set it as spacy_model in the configuration file (see 'configs/example_predict.conf'). 
```
python ./spert.py predict --config configs/example_predict.conf
```

## Notes
- To train SpERT with SciBERT \[5\] download SciBERT from https://github.com/allenai/scibert (under "PyTorch HuggingFace Models") and set "model_path" and "tokenizer_path" in the config file to point to the SciBERT directory.
- You can call "python ./spert.py train --help" / "python ./spert.py eval --help" "python ./spert.py predict --help" for a description of training/evaluation/prediction arguments.
- Please cite our paper when you use SpERT: <br/>
```
Markus Eberts, Adrian Ulges. Span-based Joint Entity and Relation Extraction with Transformer Pre-training. 24th European Conference on Artificial Intelligence, 2020.
```

## References
```
[1] Dan Roth and Wen-tau Yih, ‘A Linear Programming Formulation forGlobal Inference in Natural Language Tasks’, in Proc. of CoNLL 2004 at HLT-NAACL 2004, pp. 1–8, Boston, Massachusetts, USA, (May 6 -May 7 2004). ACL.
[2] Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh Hajishirzi, ‘Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction’, in Proc. of EMNLP 2018, pp. 3219–3232, Brussels, Belgium, (October-November 2018). ACL.
[3] Harsha Gurulingappa, Abdul Mateen Rajput, Angus Roberts, JulianeFluck,  Martin  Hofmann-Apitius,  and  Luca  Toldo,  ‘Development  of a  Benchmark  Corpus  to  Support  the  Automatic  Extraction  of  Drug-related Adverse Effects from Medical Case Reports’, J. of BiomedicalInformatics,45(5), 885–892, (October 2012).
[4] Pankaj Gupta,  Hinrich Schütze, and Bernt Andrassy, ‘Table Filling Multi-Task Recurrent  Neural  Network  for  Joint  Entity  and  Relation Extraction’, in Proc. of COLING 2016, pp. 2537–2547, Osaka, Japan, (December 2016). The COLING 2016 Organizing Committee.
[5] Iz Beltagy, Kyle Lo, and Arman Cohan, ‘SciBERT: A Pretrained Language Model for Scientific Text’, in EMNLP, (2019).
```

# Модификации https://github.com/gilmoright/spert
Запуск проводил с теми же requirements, кроме pytorch и numpy, у меня был установлен pytorch 1.10.0 (вместо 1.4.0) и numpy 1.21.4 (вместо 1.17.4)

В конфиге model_type для берта остался spert, для роберты spert-xlmroberta

Добавил класс SpERT_XLMRoberta, который является копипастой SpERT, но с другим родителем и атрибутом self.roberta вместо self.bert. Может можно было как-то элегантней это сделать - не придумал.

Роберта требует поставить SentencePiece: https://github.com/google/sentencepiece#installation

Заменил везде захардкоженные токены [CLS] на tokenizer._cls_token.

И поменял у роберты местами токены cls и pad. Потому что у неё первый имеет индекс 0, а второй 1. У берта наоборот. Этот момент очень сомнительный.

