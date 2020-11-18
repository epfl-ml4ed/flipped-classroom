## Mining interactions for early prediction of student success in flipped classrooms
[![GitHub version](https://badge.fury.io/gh/boennemann%2Fbadges.svg)](http://badge.fury.io/gh/boennemann%2Fbadges)
[![Open Source Love](https://badges.frapsoft.com/os/gpl/gpl.svg?v=102)](https://github.com/ellerbrock/open-source-badge/)

Increasingly adopted over the years, **flipped classrooms** represent a learning design that requires students to complete pre-class learning activities before participating in face-to-face sessions. For pre-class activities, this design often makes use of **videos** and **digital content** published in online platforms. The **students’ engagement** in pre-class activities is essential for the success of flipped classrooms, as these activities prepare students for effective participation in face-to-face sessions.

The widespread adoption of a flipped classroom design has spurred investigations into the issues of how to **anticipate academic performance** by analyzing digital traces, gathered from students’ interactions along pre-class activities. Student success prediction, where a model forecasts future performance of students as they interact online, is a primary while challenging goal. Trustworthy **early predictions** enable effective content personalization and adaptive teaching interventions.

This project aims at exploring behavioral patterns and predicting students’ future performance in flipped-classroom settings. As a case study, we consider a [Linear Algebra](https://www.epfl.ch/education/teaching/fr/soutien-a-lenseignement/recherche-et-developpement/exemples-de-projets/classe-inversee/) course part of the EPFL Bachelor’s Programs, delivered by Prof. [Simone Deparis](https://people.epfl.ch/simone.deparis) under a flipped-classroom approach in the [EPFL Courseware Platform](https://courseware.epfl.ch/courses/course-v1:EPFL+AlgebreLineaire+2019/course/), since 2017. The extent to which success at the end of the course can be anticipated is analyzed by examining **interaction traces left by students** while interacting with content (e.g., videos), peers (e.g., forums), and assessments (e.g., quizzes). With the support of the [Center for Digital Education (CEDE)](https://www.epfl.ch/education/educational-initiatives/cede/), the [Center for Learning Sciences (LEARN)](https://www.epfl.ch/education/educational-initiatives/home/), and the [Teaching Support Center (CAPE)](https://www.epfl.ch/education/teaching/teaching-support/who-are-we/), several indicators of engagement in online pre-class activities are being analyzed as predictors of students’ performance, and the resulting **machine-learning models** with these indicators as features are being developed.

The contributions coming from this project can shape intelligent learning platforms able to provide a **data-driven formative feedback**. These insights are essential to **assist students** in regulating their use of online learning resources and **inform teachers** on when, where and why to intervene.

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
git clone https://github.com/d-vet-ml4ed/flipped-classroom.git
cd ./flipped-classroom
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
flipped-classroom
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


