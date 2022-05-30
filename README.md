# SCDV

## Train Model

To train a model, run the following command:

```sh
python train.py \
    --data path_to_filename \
    --init_vector_type initial_vector_type \
    --vector_size dimension_of_vector \
    --num_clusters number_of_clusters \
    --save_model path_to_save_model \
    --log path_to_save_logging
```

## Load Model

```py
from scdv import SCDV
model = SCDV.load('model_path.pkl')
```

## Run IR Task

To run the information retrieval task, run the following command
```sh
python src/ir.py \
    --model path_to_model \
    --query path_to_query_file \
    --documents path_to_documents_folder \
    --output path_to_output_folder \
    --lm_ngram ngram_size_for_lm_score \
    --lambda_ lmabda_for_pv_score \
    --top k_top_results_to_return \
    --log path_to_save_logging
```

# Acknowledgements 

This project was made as a part of the Algorithms for Intelligence Web and Information Retrieval course (UE19CS332) at PES University. 

[Yashi Chawla](https://github.com/Yashi-Chawla)<br>
[Suhas Thalanki](https://github.com/thesuhas)<br>
[Vinay V](https://github.com/vinayv1102)<br>
[Anurita Bose](https://github.com/anuritabose)
