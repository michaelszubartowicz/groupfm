GroupFM
=====

Implementation and dataset of the paper

*title revealed upon successful publication*


Dataset
======
The dataset consists of four csv files found in the subfolder `GroupFM`.

*group_assignments.csv* lists which group each user belongs to.

*group_types.csv* denotes, whether the group members are familiar or collected by random selection.

*group_ratings.csv* contains the base ratings as well as the indicated impact of each contextual variable for each group.

*user_ratings.csv* contains the base ratings as well as the indicated impact of each contextual variable for each user.

Run GroupFM
=======
GroupFM is realized by a python script which is running the original libFM implementation in the background. Therefore you need to follow these steps:

- compile libFM
- depending on your platform the location string of libFM might be corrected (variable `libFM_location`)
- run the python script `Exec_GroupFM.py`

