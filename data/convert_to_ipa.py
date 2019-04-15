"""Grapheme-to-phoneme conversion script using Epitran.

Author: Shruti Rijhwani (srijhwan@cs.cmu.edu)
Last update: 2019-04-15
"""

import epitran
import sys
import codecs

filename = sys.argv[1]
epi_lang = list(sys.argv[2].split(','))
output_file = 'ipa_' + filename
epi = [epitran.Epitran(e) for e in epi_lang]

with codecs.open(filename, 'r', 'utf8') as f, codecs.open(output_file, 'w',' utf8') as out:
    for line in f:
        spl = line.strip().split(' ||| ')
        outp = [epi[i].transliterate(word) for i, word in enumerate(spl[1:-1])]
        out.write(spl[0] + ' ||| ' + ' ||| '.join(outp) + ' ||| ' + spl[-1] + '\n')
