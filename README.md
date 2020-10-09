# Flipped Classroom Project
[![GitHub version](https://badge.fury.io/gh/boennemann%2Fbadges.svg)](http://badge.fury.io/gh/boennemann%2Fbadges)
[![Dependency Status](https://david-dm.org/boennemann/badges.svg)](https://david-dm.org/boennemann/badges)
[![Open Source Love](https://badges.frapsoft.com/os/gpl/gpl.svg?v=102)](https://github.com/ellerbrock/open-source-badge/)

Over years at EPFL, first year students volunteered to participate to the Linear Algebra flipped classroom. In this experimental course, 
teaching alternates between traditional ex cathedra lectures and flipped classroom. During flipped classroom, students start at home by 
watching videos from a MOOC (Massive Open Online Course) and by practicing on simple exercises. In class, the teacher can therefore focus 
directly on the more complex exercises.

By analyzing the data collected from the MOOC, such as speed change, fast forward or when students watch the videos, we want to predict how 
students will perform and identify behavioral patterns correlated to success. 

## Table of Contents
- [Installation](#installation)
- [Folder Structure](#folder-structure) 
- [Usage](#usage)
- [Contributing](#contributing)
- [Citations](#citations)
- [License](#license)

## Installation

Clone this repository:
``` 
git clone https://github.com/mirkomarras/epfl_mooc_interactions.git
cd ./epfl_mooc_interactions
``` 

Create a Python environment:
``` 
module load python3/intel/3.6.3
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
``` 

## Folder Structure

``` 
epfl_mooc_interactions
├── config
├── data
│   ├── lin_alg_moodle
│       ├── 2017
│       ├── 2018
│       ├── problems.csv
│       ├── videos.csv
├── helpers
├── notebooks
├── tests
``` 

## Usage

...

## Contributing 

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research 
in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know.

Please feel free to file issues and pull requests on the repo and we will address them as we can.

## Citations
If you find this code useful in your work, please cite our papers:

```
```

## License
This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.


