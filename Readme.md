
## Task 1

Train model:

fasttext supervised -input task1/train.ft -output model1.ft -epoch 5 -pretrainedVectors nkjp.fasttext

Make predictions:

fasttext predict-prob model1.ft.bin task1/train.ft 2 > temp.out

Optimize threshold for F1:

cut -c 10 task1/train.ft > gold.out
python3 optimize_thresholds_fasttext_final.py gold.out temp.out 1 -w

Apply threhold:

fasttext predict-prob model1.ft.bin task1/test.ft 2 > temp.out
cut -c 10 task1/test.ft > gold.out
python3 optimize_thresholds_fasttext_final_test.py gold.out temp.out 1 `cat value`
cp test.out task1.results

## Task 2

Train model:

fasttext supervised -input task2/train.ft -output model2.ft -epoch 5 -pretrainedVectors nkjp.fasttext

Make predictions:

fasttext predict model2.ft.bin task1/test.ft  | cut -c 10- > task2.results
