# calibration dataset for base model
for i in {0..3}; do
    filename=c4-train.0000${i}-of-01024
    wget -P raw_data/base \
        https://hf-mirror.com/datasets/allenai/c4/resolve/main/en/${filename}.json.gz
    gzip -d raw_data/base/${filename}.json.gz
    mv raw_data/base/${filename}.json raw_data/base/${filename}.jsonl
done

# calibration dataset for chat model
wget -P raw_data/chat \
    https://hf-mirror.com/datasets/wql/sharegpt_cleaned_20k/resolve/main/sharegpt_cleaned_20k.jsonl