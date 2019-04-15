train_lang=am
test_lang=ti

python test.py \
--train_file "data/train_en-"$train_lang"_links" \
--model_file "models/grapheme_en-"$train_lang \
--kb "data/en_kb" \
--kb_encodings "encodings/kb" \
--links "data/en-"$train_lang"_links" \
--links_encodings "encodings/links" \
--input_file "data/test_en-"$test_lang"_links" \
--topk 30 \
--exact \
--nopivot_decode \
--pivot_decode \
--load_encodings \
--outfile "outfiles/output_"$test_lang \
--dynet-autobatching 1
