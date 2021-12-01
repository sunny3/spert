#! /bin/sh
#Путь до датасета, который содержит файлы, содержащие в названии подстроки train и test
export DS_path="./data/datasets/RDRS_multicontext_train_filtered_nltk_tokenization_raw/folds/6/"
#Путь до директории сохранения датасета в конвертированном сперт формате и сохранения результатов с предиктами, должен быть относительным путем к папке ./data/datasets
export RES_path="RDRS_debug/folds/6/"
#Путь до модели, можно использовать и веб-путь. Pipeline работает с моделями следующих топологий: xlm-roberta-base и xlm-roberta-large 
export MODEL_path="xlm-roberta-large"


python ./converters/convert_input.py --data_path $DS_path --res_path $RES_path | tee ./input_convertion_report.txt
export CONFIG=`grep .conf ./input_convertion_report.txt`
export DS_name=`tac ./input_convertion_report.txt |egrep -m 1 .`
export ORIGIN_TEST=`grep -m2 test ./input_convertion_report.txt | tail -n-1`
rm ./input_convertion_report.txt
CONFIG="./configs/"$CONFIG
export Epoch_num=`grep epochs $CONFIG | grep -Eo '[0-9]{1,}'`
sed -i 's|xlm-roberta-large|'$MODEL_path'|g' $CONFIG

python spert.py train --config $CONFIG

export LAST_exp=`ls ./data/log/$DS_name -t | egrep -m 1 .`
echo take predictions from ./data/log/$DS_name/$LAST_exp
cp ./data/log/$DS_name/$LAST_exp/predictions_valid_epoch_$Epoch_num.json ./data/datasets/$RES_path

python ./converters/convert_pred.py --data_path ./data/datasets/$RES_path
python spert_eval.py -t $ORIGIN_TEST -p ./data/datasets/$RES_path/pred.json -o ./data/datasets/$RES_path/
