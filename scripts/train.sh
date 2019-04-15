# to train with CPU, remove the --dynet-gpu flag
lang=am

python train.py --train_file "data/train_en-"$lang"_links" \
--val_file "data/val_en-"$lang"_links" \
--model_file "models/grapheme_en-"$lang \
--dynet-gpu --dynet-autobatching 1 --dynet-mem 2000
