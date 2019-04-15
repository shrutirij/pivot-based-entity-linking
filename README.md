#  Pivot-based Entity Linking
  
Pivot-based entity linking (PBEL) is an entity linking method that uses high-resource languages as intermediates while doing low-resource cross-lingual entity linking. The details are described in the paper: [Zero-shot Neural Transfer for Cross-lingual Entity Linking](https://arxiv.org/abs/1811.04154).

## Dependencies
* Python3 with [NumPy](http://www.numpy.org)
* [Dynet](https://github.com/clab/dynet)
* [PanPhon](https://github.com/dmort27/panphon)
* [Epitran](https://github.com/dmort27/epitran) for converting strings to IPA

## Data

* Download sample data for one language pair [here](https://drive.google.com/file/d/1T3QDE9eMmbXHJbSmb_tnTomqqugUzNaL/view?usp=sharing).
* Download data for all 54 training languages and 9 test languages [here](https://drive.google.com/file/d/1-5ss6miJxCLEX3Dcxt0sK_4j_0NAYrYw/view?usp=sharing).

## Usage
* ```train.py``` is used for training the entity similarity model.
    * Default values for hyperparameters are in the script -- 64 size character embeddings and 1024 hidden size for the character LSTM.
* ```test.py``` is used both for encoding the knowledge base as well as retrieving entity linking candidates for test data or other input files.
* See the ```scripts/``` folder for examples using data from [here](https://drive.google.com/file/d/1T3QDE9eMmbXHJbSmb_tnTomqqugUzNaL/view?usp=sharing).

## References
If you use this repository, please cite
```
@inproceedings{rijhwani19aaai,
    title = {Zero-shot Neural Transfer for Cross-lingual Entity Linking},
    author = {Shruti Rijhwani and Jiateng Xie and Graham Neubig and Jaime Carbonell},
    booktitle = {Thirty-Third AAAI Conference on Artificial Intelligence (AAAI)},
    address = {Honolulu, Hawaii},
    month = {January},
    url = {https://arxiv.org/abs/1811.04154},
    year = {2019}
}
```
