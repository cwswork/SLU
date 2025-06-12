# LCA-UEA

Source code and datasets for paper: 
[***Learnable Convolutional Attention Network for Unsupervised Knowledge Graph Entity Alignment***]

## Datasets

> Please first download the main datasets [here](https://www.jianguoyun.com/p/DY8iIAsQ2t_lCBjK3oUEIAA) 
, path datasets [here](https://www.jianguoyun.com/p/DWzhBksQ2t_lCBjon78EIAA)
and extract them into `datasets/` directory.

Initial datasets WN31-15K and DBP-15K are from [OpenEA](https://github:com/nju-websoft/OpenEA) and [JAPE](https://github.com/nju-websoft/JAPE).

Initial datasets DWY100K is from  [BootEA](https://github.com/nju-websoft/BootEA).

Take the dataset zh_en(DBP15K) as an example, the main datasets contains:
* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* triples_1(triples_2): relation triples encoded by ids;
* LaBSE_emb_1(LaBSE_emb_2): the input entity name feature matrix initialized by word vectors;
* link/test_links(train_links/valid_links): entity links encoded by ids;

## Environment

* Python>=3.10
* pytorch>=1.7.0
* Numpy
* json


## Running

To run LCA-UEA model on all dataset, use the following script:
```
python3 align/exc.py
```

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.
> If you have any difficulty or question in running code and reproducing expriment results, please email to cwswork@qq.com.

## Citation

If you use this model or code, please cite it as follows:

*Weishan Cai, Wenjun Ma*, 
“Learnable Convolutional Attention Network for Unsupervised Knowledge Graph Entity Alignment”
