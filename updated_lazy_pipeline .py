import time
from typing import Iterator, List, Collection, Callable
from collections import Counter
import pandas as pd
from tqdm import tqdm
import math

# A very general type hint for a prediction function.
# A prediction function takes a triplet of (test object description, train descriptions, train labels)
# and outputs a bool prediction
PREDICTION_FUNCTION_HINT = Callable[
    [Collection, Collection[Collection], Collection[bool]], bool
]
def comapre(intsec, example, categorical_columns, numerical_columns):
    for cat_c in categorical_columns:
        if intsec[cat_c][0] != 'Any' and intsec[cat_c][0] != example[cat_c] :
            return False
    for num_c in numerical_columns:
        if example[num_c] > intsec[num_c][0][1] or example[num_c] < intsec[num_c][0][0]:
            return False
    return True

def compute_intsec(a, b, empty_df, categorical_columns, numerical_columns):
    intsec = pd.DataFrame(data=empty_df) 
    for cat_c in categorical_columns:
        if a[cat_c] == b[cat_c]:
            intsec[cat_c] = a[cat_c]
        else: 
            intsec[cat_c] = 'Any' 
    for num_c in numerical_columns:
        intsec[num_c][0] = [min(a[num_c], b[num_c]), max(a[num_c], b[num_c])]

    return intsec  

def load_data(df_name: str) -> pd.DataFrame:
    """Generalized function to load datasets in the form of pandas.DataFrame"""
    if df_name == 'tic_tac_toe':
        return load_tic_tac_toe()

    raise ValueError(f'Unknown dataset name: {df_name}')


def load_tic_tac_toe() -> pd.DataFrame:
    """Load tic-tac-toe dataset from UCI repository"""
    column_names = [
        'top-left-square', 'top-middle-square', 'top-right-square',
        'middle-left-square', 'middle-middle-square', 'middle-right-square',
        'bottom-left-square', 'bottom-middle-square', 'bottom-right-square',
        'Class'
    ]
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data'
    df = pd.read_csv(url, names=column_names)
    df['Class'] = [x == 'positive' for x in df['Class']]
    return df


def binarize_X(X: pd.DataFrame) -> 'pd.DataFrame[bool]':
    """Scale values from X into pandas.DataFrame of binary values"""
    dummies = [pd.get_dummies(X[f], prefix=f, prefix_sep=': ') for f in X.columns]
    X_bin = pd.concat(dummies, axis=1).astype(bool)
    return X_bin

def improved_predict_with_generators(
        x: set, X_train: List[set], Y_train: List[bool], 
        min_cardinality: int = 1
) -> bool:
    X_pos = []
    X_neg = []
    
    X_neg_for_pos = {} # create two hash maps
    X_pos_for_neg = {}
    
    idx = 0
    for x_train, y in zip(X_train, Y_train): # Distribute data into lists
        if y:
            X_pos.append(x_train)
            for val in x_train:
                if X_pos_for_neg.get(val) == None:
                    X_pos_for_neg[val] = [idx]
                else:
                    X_pos_for_neg[val].append(idx)
        else:
            X_neg.append(x_train)
            for val in x_train:
                if X_neg_for_pos.get(val) == None:
                    X_neg_for_pos[val] = [idx]
                else:
                    X_neg_for_pos[val].append(idx)
        idx += 1

    n_counters_pos = 0  
    for x_pos in X_pos: # begin to compare with positive examples
        temp = []
        intersection_pos = x & x_pos # counting intersection
        if len(intersection_pos) < min_cardinality: 
            continue
        
        o = 0
        # to count something as counterexample all attributes from intersec must correspond to at least one common object. 
        for pos_atr_from_intsec in intersection_pos: # we find common objects for all attributes from intersection
            if X_neg_for_pos.get(pos_atr_from_intsec) != None: 
                if o == 0:
                    final_set = set(X_neg_for_pos.get(pos_atr_from_intsec))
                    o = 1
                else:
                    final_set = final_set & set(X_neg_for_pos.get(pos_atr_from_intsec))
            else: 
                break
        else:    
            n_counters_pos += len(final_set)
    # same
    n_counters_neg = 0  
    for x_neg in X_neg:
        temp = []
        intersection_neg = x & x_neg 
        if len(intersection_neg) < min_cardinality: 
            continue
            
        o = 0
        for neg_atr_from_intsec in intersection_neg:
            if X_pos_for_neg.get(neg_atr_from_intsec) != None:
                if o == 0:
                    final_set = set(X_pos_for_neg.get(neg_atr_from_intsec))
                    o = 1
                else:
                    final_set = final_set & set(X_pos_for_neg.get(neg_atr_from_intsec))
            else: 
                break
        else:    
            n_counters_neg += len(final_set)
    

    perc_counters_pos = n_counters_pos / (n_counters_pos + n_counters_neg)
    perc_counters_neg = n_counters_neg / (n_counters_pos + n_counters_neg)
    
    prediction = perc_counters_pos < perc_counters_neg
    return prediction

def predict_with_generators(
        x: set, X_train: List[set], Y_train: List[bool],
        min_cardinality:int = 1
) -> bool:
    """Lazy prediction for ``x`` based on training data ``X_train`` and ``Y_train``

    Parameters
    ----------
    x : set
        Description to make prediction for
    X_train: List[set]
        List of training examples
    Y_train: List[bool]
        List of labels of training examples
    min_cardinality: int
        Minimal size of an intersection required to count for counterexamples

    Returns
    -------
    prediction: bool
        Class prediction for ``x`` (either True or False)
    """
    X_pos = [x_train for x_train, y in zip(X_train, Y_train) if y]
    X_neg = [x_train for x_train, y in zip(X_train, Y_train) if not y]

    n_counters_pos = 0  # number of counter examples for positive intersections
    for x_pos in X_pos:
        intersection_pos = x & x_pos
        if len(intersection_pos) < min_cardinality:  # the intersection is too small
            continue

        for x_neg in X_neg:  # count all negative examples that contain intersection_pos
            if (intersection_pos & x_neg) == intersection_pos:
                n_counters_pos += 1
    n_counters_neg = 0  # number of counter examples for negative intersections
    for x_neg in X_neg:
        intersection_neg = x & x_neg
        if len(intersection_neg) < min_cardinality:
            continue

        for x_pos in X_pos:  # count all positive examples that contain intersection_neg
            if (intersection_neg & x_pos) == intersection_neg:
                n_counters_neg += 1
                
    perc_counters_pos = n_counters_pos / len(X_pos)
    perc_counters_neg = n_counters_neg / len(X_neg)

    prediction = perc_counters_pos < perc_counters_neg
    return prediction


def improved_predict_array(
        X: List[set], Y: List[bool],
        n_train: int, update_train: bool = True, use_tqdm: bool = False,
        predict_func: PREDICTION_FUNCTION_HINT = improved_predict_with_generators
) -> Iterator[bool]:
    """Predict the labels of multiple examples from ``X``

    Parameters
    ----------
    X: List[set]
        Set of train and test examples to classify represented with subsets of attributes
    Y: List[bool]
        Set of train and test labels for each example from X
    n_train: int
        Initial number of train examples. That is, make predictions only for examples from X_train[n_train:]
    update_train: bool
        A flag whether to consider true labels of predicted examples as training data or not.
        If True, then for each X_i the training data consists of X_1, X_2, ..., X_{n_train}, ...,  X_{i-1}.
        If False, then for each X_i the training data consists of X_1, X_2, ..., X_{n_train}
    use_tqdm: bool
        A flag whether to use tqdm progress bar (in case you like progress bars)
    predict_func: <see PREDICTION_FUNCTION_HINT defined in this file>
        A function to make prediction for each specific example from ``X``.
        The default prediction function is ``predict_with_generator`` (considered as baseline for the home work).

    Returns
    -------
    prediction: Iterator
        Python generator with predictions for each x in X[n_train:]
    """
    for i, x in tqdm(
        enumerate(X[n_train:]),
        initial=n_train, total=len(X),
        desc='Predicting step by step',
        disable=not use_tqdm,
    ):
        n_trains = n_train + i if update_train else n_train
        yield predict_func(x, X[:n_trains], Y[:n_trains])
        
def predict_array(
        X: List[set], Y: List[bool],
        n_train: int, update_train: bool = True, use_tqdm: bool = False,
        predict_func: PREDICTION_FUNCTION_HINT = predict_with_generators
) -> Iterator[bool]:
    """Predict the labels of multiple examples from ``X``

    Parameters
    ----------
    X: List[set]
        Set of train and test examples to classify represented with subsets of attributes
    Y: List[bool]
        Set of train and test labels for each example from X
    n_train: int
        Initial number of train examples. That is, make predictions only for examples from X_train[n_train:]
    update_train: bool
        A flag whether to consider true labels of predicted examples as training data or not.
        If True, then for each X_i the training data consists of X_1, X_2, ..., X_{n_train}, ...,  X_{i-1}.
        If False, then for each X_i the training data consists of X_1, X_2, ..., X_{n_train}
    use_tqdm: bool
        A flag whether to use tqdm progress bar (in case you like progress bars)
    predict_func: <see PREDICTION_FUNCTION_HINT defined in this file>
        A function to make prediction for each specific example from ``X``.
        The default prediction function is ``predict_with_generator`` (considered as baseline for the home work).

    Returns
    -------
    prediction: Iterator
        Python generator with predictions for each x in X[n_train:]
    """
    for i, x in tqdm(
        enumerate(X[n_train:]),
        initial=n_train, total=len(X),
        desc='Predicting step by step',
        disable=not use_tqdm,
    ):
        n_trains = n_train + i if update_train else n_train
        yield predict_func(x, X[:n_trains], Y[:n_trains])

def apply_stopwatch(iterator: Iterator):
    """Measure run time of each iteration of ``iterator``

    The function can be applied e.g. for the output of ``predict_array`` function
    """
    outputs = []
    times = []

    t_start = time.time()
    for out in iterator:
        dt = time.time() - t_start
        outputs.append(out)
        times.append(dt)
        t_start = time.time()

    return outputs, times
