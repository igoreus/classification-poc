train.sh#!/usr/bin/env bash

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " "
}

DATA_DIR=data/fasttext
RESULT_DIR=result/fasttext
CSV_DATA=${DATA_DIR}/data.csv
CSV_TEST_DATA=${DATA_DIR}/data_test.csv

mkdir -p "${DATA_DIR}"
mkdir -p "${RESULT_DIR}"

if [ ! -f "${DATA_DIR}/data.train" ] || [ "$1" == "force" ]
then
  python create_csv_from_db.py ${CSV_DATA} ${CSV_TEST_DATA}
  cat "${CSV_DATA}" | normalize_text > "${DATA_DIR}/data.train"
  cat "${CSV_TEST_DATA}" | normalize_text > "${DATA_DIR}/data.test"
fi

./fasttext supervised -input "${DATA_DIR}/data.train" -output "${RESULT_DIR}/data" -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 10 -thread 4
./fasttext test "${RESULT_DIR}/data.bin" "${DATA_DIR}/data.test"
./fasttext predict "${RESULT_DIR}/data.bin" "${DATA_DIR}/data.test" > "${RESULT_DIR}/data.test.predict"

python test_prediction.py ${CSV_TEST_DATA} "${RESULT_DIR}/data.test.predict"
