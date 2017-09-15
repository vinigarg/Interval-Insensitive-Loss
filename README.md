# Interval-Insensitive-Loss
Performed ordinal classification on partially annotated examples.

## Goal
Goal was to estimate the age (inter- val/range) of a person’s face.
1. A numbered list

## Structure of Code
The following steps include the basic structure of code :
1. Read the entire data from the file.
2. Calculate the total number of classes or bins based on the provided bin size and size of data.
number_of_classes = data_size / bin_size
3. Define the ranges (left extreme and right extreme) for each bin.
4. Segregate the data into files as “train” and “test” using Kfold algorithm.
5. Calculate values of different variables in the dual form and provide it to QP solver to find
the Lagrangian constants .
6. Find weight vector using Lagrangian constants.
7. Classify the input samples using the obtained weight vector and determine accuarcy and
mean absolute error loss.
8. Plot the graph based on the above results.

There are two files namely : binning.py and computation.py
binning.py : steps 1 to 4
computation.py : steps 5 to 8

## Execute the code
python computation.py bin_size
Example: python computation.py 50

## Output Format
    bin_size
    accuracy : case_1 case_2
    MAE : case_1 case_2
    
Example:

    10
    Accuracy :  0.00963995354239 0.0720092915215
    MAE :  89.9709639954 89.4425087108

    50
    Accuracy :  0.349361207898 0.49756097561
    MAE :  69.6225319396 63.8850174216

    100
    Accuracy :  0.473403019744 0.583739837398
    MAE :  54.6109175377 49.837398374

    150
    Accuracy :  0.564576074332 0.655400696864
    MAE :  42.4390243902 34.5644599303





## Refernce 
http://proceedings.mlr.press/v39/antoniuk14.pdf
