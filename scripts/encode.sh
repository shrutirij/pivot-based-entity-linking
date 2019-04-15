# to encode with CPU, remove the --dynet-gpu flag
train_lang=am
test_lang=ti

python test.py \
--train_file "data/en-"$train_lang"_links" \
--model_file "models/grapheme_en-"$train_lang \
--kb "data/en_kb" \
--kb_encodings "encodings/kb" \
--links "data/en-"$train_lang"_links" \
--links_encodings "encodings/links" \
--dynet-gpu --dynet-mem 2000 --dynet-autobatching 1
