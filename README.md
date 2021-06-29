## Can Feature Predictive Power Generalize? Benchmarking Early Predictors of Student Success across Flipped and Online Courses

This repository is the official implementation of the EDM 2021 entitled "[Can Feature Predictive Power Generalize? Benchmarking Early Predictors of Student Success across Flipped and Online Courses](https://youtu.be/_1sdX3W5Q5A)". 

![Our approach](assets/schema.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To prepare data:



## Feature Extraction

To extract a setof features for a course, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Training and Evaluation

To train and evaluate a predictor on a set of features, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Contributing 

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research 
in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know.

Please feel free to file issues and pull requests on the repo and we will address them as we can.

## Citations
If you find this code useful in your work, please cite our papers:

```
Marras, M., Vignoud, J., Käser, T. (2021). 
Can Feature Predictive Power Generalize? Benchmarking Early Predictors of Student Success across Flipped and Online Courses. 
In: Proceedings of the 14th International Conference on Educational Data Mining (EDM 2021). 
```

## License
This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.


