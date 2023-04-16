Dataset is provided by [CodeXGlue](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token) project available on GitHub

## About dataset
From [CodeXGlue](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token) project.
### Github Java Corpus
We use java corpus dataset mined by Allamanis and Sutton, in their MSR 2013 paper [Mining Source Code Repositories at Massive Scale using Language Modeling](https://homepages.inf.ed.ac.uk/csutton/publications/msr2013.pdf). We follow the same split and preprocessing in Karampatsis's ICSE 2020 paper [Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code](http://homepages.inf.ed.ac.uk/s1467463/documents/icse20-main-1325.pdf).

To download the preprocessed dataset, navigate to `dataset/javaCorpus` directory, and run
```shell
bash download.sh
python preprocess.py --base_dir=token_completion --output_dir=token_completion
```

### Data Preprocessing

- Tokenization is applied since we focus the token-level completion. 
- We normalize uncommon literals for better user experience. Developers sometimes leave their names, IP address, phone numbers in their codes, and we don't encourage models to focus on these string or numeric literals. So we normalized these literals by some special tokens. Considering that frequently used literals may contain useful information, e.g. "\_\_main\_\_" or "utf-8", we preserve the 200 most frequent string and 30 most frequent numeric literals. These literals will be normalized by tokens in `"<STR_LIT:utf-8>"` format, while uncommon literals are replaced by `<STR_LIT>` or `<NUM_LIT>`. 
- We add `<s>` and `</s>` to indicate the start and the end of one piece of code. 
- `<EOL>` is added in python corpus to mark the ending of a line since in python there is no `;` or `}` to mark the ending of a statement like in java.

### Data Format

Code corpus are saved in txt format files. one line is a tokenized code snippets:
```
<s> from __future__ import unicode_literals <EOL> from django . db import models , migrations <EOL> class Migration ( migrations . Migration ) : <EOL> dependencies = [ <EOL> ] <EOL> operations = [ <EOL> migrations . CreateModel ( <EOL> name = '<STR_LIT>' , <EOL> fields = [ <EOL> ( '<STR_LIT:id>' , models . AutoField ( verbose_name = '<STR_LIT>' , serialize = False , auto_created = True , primary_key = True ) ) , <EOL> ( '<STR_LIT:name>' , models . CharField ( help_text = b'<STR_LIT>' , max_length = <NUM_LIT> ) ) , <EOL> ( '<STR_LIT:image>' , models . ImageField ( help_text = b'<STR_LIT>' , null = True , upload_to = b'<STR_LIT>' , blank = True ) ) , <EOL> ] , <EOL> options = { <EOL> '<STR_LIT>' : ( '<STR_LIT:name>' , ) , <EOL> '<STR_LIT>' : '<STR_LIT>' , <EOL> } , <EOL> bases = ( models . Model , ) , <EOL> ) , <EOL> ] </s>
```


### Data Statistics

Data statistics of Github Java Corpus dataset are shown in the below table:

| Data Split  |   #Files   |   #Tokens   |
| ----------- | :--------: | :---------: |
|    Train    |   12,934   |    15.7M    |
|     Dev     |    7,176   |     3.8M    |
|    Test     |    8,268   |     5.3M    |
