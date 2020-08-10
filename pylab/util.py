'''Utility functions used by multiple modules
'''
import matplotlib.pyplot as plt
import numpy as np
import pickle
import inspect
import logging
import random
import scipy.sparse
import numba
from timeit import default_timer
import itertools
import re
import ete3

import os
from pathlib import Path

import scipy.spatial

# Constants
NAME_FORMATTER = '%(name)s'
ID_FORMATTER = '%(id)s'
INDEX_FORMATTER = '%(index)s'
SPECIES_FORMATTER = '%(species)s'
SPECIESX_FORMATTER = '%(speciesX)s'
GENUS_FORMATTER = '%(genus)s'
FAMILY_FORMATTER = '%(family)s'
CLASS_FORMATTER = '%(class)s'
ORDER_FORMATTER = '%(order)s'
PHYLUM_FORMATTER = '%(phylum)s'
KINGDOM_FORMATTER = '%(kingdom)s'
LCA_FORMATTER = '%(lca)s'

_TAXLEVELS = ['species', 'genus', 'family', 'class', 'order', 'phylum', 'kingdom']
_TAXFORMATTERS = ['%(species)s', '%(genus)s', '%(family)s', '%(class)s', '%(order)s', '%(phylum)s', '%(kingdom)s']
_SPECIESX_SEARCH = re.compile('\%\(species[0-9]+\)s')

def issquare(a):
    '''Checks if the input array is a square

    Parameters
    ----------
    a : any
        Instance we are checking
    
    Returns
    -------
    bool
        True if `a` is a square matrix
    '''
    try:
        return (a.shape[0] == a.shape[1]) and (len(a.shape) == 2)
    except:
        return False

def isbool(a):
    '''Checks if `a` is a bool

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a bool
    '''
    return a is not None and np.issubdtype(type(a), np.bool_)

def isint(a):
    '''Checks if `a` is an int

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is an int
    '''
    return a is not None and np.issubdtype(type(a), np.integer)

def isfloat(a):
    '''Checks if `a` is a float

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a float
    '''
    return a is not None and np.issubdtype(type(a), np.floating)

def iscomplex(a):
    '''Checks if `a` is a complex number

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a complex number
    '''
    return a is not None and np.issubdtype(type(a), np.complexfloating)

def isnumeric(a):
    '''Checks if `a` is a float or an int - (cannot be a bool)

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a float or an int.
    '''
    return a is not None and np.issubdtype(type(a), np.number)

def isarray(a):
    '''Checks if `a` is an array

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is an array
    '''
    return (type(a) == np.ndarray or type(a) == list or \
        scipy.sparse.issparse(a)) and a is not None

def isstr(a):
    '''Checks if `a` is a str

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a str
    '''
    return a is not None and type(a) == str

def istype(a):
    '''Checks if `a` is a Type object

    Example
    -------
    >>> istype(5)
    False
    >>> istype(float)
    True

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a tuple
    '''
    return type(a) == type

def istuple(a):
    '''Checks if `a` is a tuple object

    Example
    -------
    >>> istuple(5)
    False
    >>> istuple((5,))
    True

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a tuple
    '''
    return type(a) == tuple

def isdict(a):
    '''Checks if `a` is a dict object

    Example
    -------
    >>> isdict(5)
    False
    >>> isdict({5:2})
    True

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a dict
    '''
    return type(a) == dict

def istree(a):
    '''Checks if `a` is an ete3 Tree object

    Example
    -------
    >>> istree(5)
    False
    >>> a = ete3.Tree("((a,b),c);")
    >>> istree(a)
    True

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a ete3 Tree
    '''
    return type(a) == ete3.Tree

def itercheck(xs, f):
    '''Checks every element in xs with f and returns an array
    for each entry

    Parameters
    ----------
    xs : array_like(any)
        - A list of instances to check the type of
    f : callable
        - Type checking function
    Returns
    -------
    list(bool)
        Checks for each element in the `xs`
    '''
    return [f(x) for x in xs]

def asvname_formatter(format, asv, asvs, lca=True):
    '''Format the label of an ASV. Specify the ASV by
    it's index in the ASVSet `asvs`

    Example:
        asv is an ASV object at index 0 where
        asv.genus = 'A'
        asv.id = 1234532

        asvname_formatter(
            format='%(genus)s: %(index)s',
            asv=1234532,
            asvs=asvs)
        >>> 'A: 0'

        asvname_formatter(
            format='%(genus)s: %(genus)s',
            asv=1234532,
            asvs=asvs)
        >>> 'A: A'

        asvname_formatter(
            format='%(index)s',
            asv=1234532,
            asvs=asvs)
        >>> '0'

        asvname_formatter(
            format='%(geNus)s: %(genus)s',
            asv=1234532,
            asvs=asvs)
        >>> '%(geNus)s: A'

    Parameters
    ----------
    format : str
        This is the format for us to do the labels
        Formatting options:
            '%(name)s'
                Name of the ASV (pylab.base.ASV.name)
            '%(id)s'
                ID of the ASV (pylab.base.ASV.id)
            '%(index)s'
                The order that this appears in the ASVSet
            '%(species)s'
                `'species'` taxonomic classification of the ASV
            '%(speciesX)s'
                `'species'` taxonomic classification of the ASV for only up to the first 
                `X` spceified
            '%(genus)s'
                `'genus'` taxonomic classification of the ASV
            '%(family)s'
                `'family'` taxonomic classification of the ASV
            '%(class)s'
                `'class'` taxonomic classification of the ASV
            '%(order)s'
                `'order'` taxonomic classification of the ASV
            '%(phylum)s'
                `'phylum'` taxonomic classification of the ASV
            '%(kingdom)s'
                `'kingdom'` taxonomic classification of the ASV
            '%(lca)s'
                Least common ancestor. If species is 'NA', then it will go to family.
                It will keep travelling up the tree until it finds something not nan.
                Example:
                asv is an ASV object at index 0 where
                asv.genus = 'nan'
                asvs.family = 'B'
                asv.id = 1234532

    asv : str, int, ASV
        Either the ASV or an id for the ASV
    asvs : pylab.base.ASVSet
        Dataset containing all of the information for the ASVs
    lca : bool
        If True and the specified taxonomic level is not specified (nan), then
        we substitute it with the least common ancestor up from the current level

    '''
    asv = asvs[asv]
    index = asv.idx

    label = format.replace(NAME_FORMATTER, str(asv.name))
    label = label.replace(ID_FORMATTER, str(asv.id))
    label = label.replace(INDEX_FORMATTER,  str(index))

    
    # Replcate speciesX formatter
    X = _SPECIESX_SEARCH.search(format)
    if X is not None:
        while True:
            X = X[0]
            n = int(X.replace('%(species', '').replace(')s',''))
            try:
                a = '/'.join(asv.get_taxonomy('species').split('/')[:n])
            except:
                a = 'nan'
            format = format.replace(X,a)
            X = _SPECIESX_SEARCH.search(format)
            if X is None:
                break
    
    for i in range(len(_TAXLEVELS)):
        taxlevel = _TAXLEVELS[i]
        fmt = _TAXFORMATTERS[i]
        try:
            label = label.replace(fmt, str(asv.get_taxonomy(taxlevel, lca=False)))
        except:
            logging.critical('asv: {}'.format(asv))
            logging.critical('fmt: {}'.format(fmt))
            logging.critical('label: {}'.format(label))
            raise

    if LCA_FORMATTER in label:
        lineage = list(asv.get_lineage(level='species'))
        while len(lineage) > 0:
            if str(lineage[-1]) != 'nan':
                label = label.replace(LCA_FORMATTER, str(lineage[-1]))
                break
            else:
                lineage = lineage[:-1]
        if len(lineage) == 0:
            logging.warning('All taxonomic levels are nans: {}'.format(asv.get_lineage(level='species')))

    return label

@numba.jit(nopython=True, cache=True)
def fast_index(M, rows, cols):
    '''Fast index fancy indexing the matrix M. ~98% faster than regular
    fancy indexing
    M MUST BE C_CONTIGUOUS for this to actually help.
        If it is not C_CONTIGUOUS then ravel will have to copy the 
        array before it flattens it. --- super slow

    Parameters
    ----------
    M : np.ndarray 2-dim
        Matrix we are indexing at 2 dimensions
    rows, cols : np.ndarray
        rows and columns INDEX arrays. This will not work with bool arrays

    Returns
    -------
    np.ndarray
    '''
    return (M.ravel()[(cols + (rows * M.shape[1]).reshape(
        (-1,1))).ravel()]).reshape(rows.size, cols.size)

def toarray(x, dest=None, T=False):
    '''Converts `x` into a C_CONTIGUOUS numpy matrix if 
    the matrix is sparse. If it is not sparse then it just returns
    the matrix.

    Parameters
    ----------
    x : scipy.sparse, np.ndarray
        Array we are converting
    dest : np.ndarray
        If this is specified, send the array into this array. Assumes
        the shapes are compatible. Else create a new array
    T : bool
        If True, set the transpose

    Returns
    -------
    np.ndarray
    '''
    if scipy.sparse.issparse(x):
        if dest is None:
            if T:
                dest = np.zeros(shape=x.T.shape, dtype=x.dtype)
            else:
                dest = np.zeros(shape=x.shape, dtype=x.dtype)
        if T:
            x.T.toarray(out=dest)
        else:
            x.toarray(out=dest)
        return dest
    else:
        if T:
            return x.T
        else:
            return x

def subsample_timeseries(T, sizes, approx=True):
    '''Subsample the time-series `T` into size in `sizes`. Note that
    this algorithm does not guarentee any timepoints to stay.

    The baseline algorithm calculates:
    d(R) = \sum_{i \neq j} 1 / (R[i] - R[j])^2
    for every *COMBINATION* of T of size size. If |T| = 75 and size = 45,
    then there are 45! * (75 - 45)! = 3.2e88 combinations - this is not 
    doable.

    Approximation
    -------------
    To approximate this, we divide the current size of the of the elements 
    into `size+1` (approximately) equal intervals
    Example:
        T = [0, 0.5, 1, 2, 4, 4.5, 5, 7, 8, 10]
        len(T) = 10
        sizes = [8, 6]

        time_indexes for T:
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9

        Make 8 point interval:
            Need to delete 2 time points,
            10 - 8 = 2
            floor(10 / 3) = 3
            Delete every 3 elements
            0 , 1 , 3, 4, 6, 7, 9, 10

            T_8 = []
    

        
        
        

    Parameters
    ----------
    T : array_like
        These are the times that we want to subsample
    sizes : int, array(int)
        These are the sizes we want to subsample to. 

    Returns
    -------
    list(np.ndarray(sizes))
        A list of time series for each size in decreasing size order
    '''
    if not isarray(T):
        raise TypeError('`T` ({}) must be an array'.format(type(T)))
    if isint(sizes):
        sizes = [sizes]
    elif isarray(sizes):
        for size in sizes:
            if not isint(size):
                raise TypeError('Each size in `sizes` ({}) must be an int ({})'.format(
                    type(size), sizes))
    else:
        raise ValueError('`sizes` ({}) must be an int or an array f ints'.format(type(sizes)))

    T = np.asarray(T)
    sizes = np.unique(np.asarray(sizes, dtype=int))
    sizes[::-1].sort()
    ret = []
    l = len(T)

    prev_tp = np.arange(len(T))
    for n in sizes:
        spacings = []
        subsets = []

        for subset in itertools.combinations(prev_tp, n):
            subsets.append(subset)
            subset = list(subset)
            subset = [-1] + subset + [l+1]
            subset = np.array(subset)
            spacings.append((1/np.power(scipy.spatial.distance.pdist(
                subset[:, np.newaxis]), 2)).sum())

        idxs = np.array(subsets[spacings.index(min(spacings))])
        ret.append(T[idxs])
        prev_tp = idxs

    return ret

def coarsen_phylogenetic_tree(tree, depth):
    '''Coarsen the tree to the maximum depth `depth`.

    This is a recursive algorithm:
        We stop on two conditions:
            when 

    Parameters
    ----------
    tree : ete3.Tree
        Phylogenetic tree
    depth : float
        Depth to coarsen to

    Returns
    -------
    ete3.Tree, dict
        ete3.Tree : is the coarsened tree
        dict : is a dictionary mapping the name of the new leaves
            to the ASVs that are contained within it
    '''
    if not istree(tree):
        raise TypeError('`tree` ({}) must be of type tree'.format(type(tree)))
    if not isnumeric(depth):
        raise TypeError('`depth` ({}) must be numeric'.format(type(depth)))
    if depth < 0:
        raise ValueError('`depth` ({}) must be >= 0'.format(depth))

    tree = _coarsen_loop(tree=tree, root=tree, depth=depth)

    # Rename all of the leaves and make a mapping
    mapping = {}
    for node_name, leaf in enumerate(tree.get_leaves()):
        mapping[str(node_name)] = leaf.name.split('|')
        leaf.name = str(node_name)

    return tree, mapping

def _coarsen_loop(tree, root, depth):
    '''Set the temporary name for a cluster to be:
        'name1|name2|...' where name1... are the 
        children of the node
    '''
    children = tree.get_children()
    if len(children) == 0:
        # This is a leaf node
        return tree

    for child in children:
        # Check the childs distance from the root
        dist = root.get_distance(child)
        dist_node = tree.get_distance(child)
        if dist < depth:
            # We are not at the desired depth yet, keep recurring
            child2 = _coarsen_loop(tree=child, root=root, depth=depth)
            tree.remove_child(child)
            tree.add_child(child2, dist=dist_node)
        else:
            # We have hit the threshold for the depth, set all children
            # as a new cluster and the name
            name = '|'.join(child.get_leaf_names())

            # Make the distance `depth`
            tree_dist = root.get_distance(tree)
            dist_node = np.min([dist_node, depth-tree_dist])

            child2 = ete3.Tree(name=name)
            tree.remove_child(child)
            tree.add_child(child2, dist=dist_node)

    return tree

class inspect_trace:
    '''This is a decorator that logs the trace of a function.

    If the function throws an error when it is being called, it will produce the array 
        [(file_name, line_no, function), ...]
    where it produces the file name, line number, and function that was called, and then
    it will throw the exception.
    The functions are in order for how close they are to the error.

    Example: (temp.py file)
        1 def foo():
        2     bar()
        3
        4 def bar():
        5     mer()
        6
        7 def mer():
        8     raise IndexError(...)
        9
        10 foo()
        CRITICAL: [('temp.py', 8, 'mer'), ('temp.py', 5, 'bar'), ('temp.py', 2, 'foo'), ('temp.py', 10, '<module>')]
        IndexError

    Parameters
    ----------
    max_trace : int, Optional
        If Specified it will only trace up to the number specified. Otherwise it will trace up to the module
    '''

    def __init__(self, max_trace=None):
        if max_trace is not None:
            if not isint(max_trace):
                raise TypeError('`max_trace` ({}) must be an int'.format(type(max_trace)))
            if max_trace <= 0:
                raise TypeError('`max_trace` ({}) must be >= 1'.format(max_trace))
        self.max_trace = max_trace

    def __call__(self, f):
        def wrapped_func(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except:
                stack = inspect.stack()
                trace = []
                if self.max_trace is not None:
                    l = self.max_trace
                else:
                    l = len(stack)
                for i in range(1, l):
                    trace.append((stack[i][1].split('/')[-1], stack[i][2], stack[i][3]))
                logging.critical('Error thrown in "{}". Trace: {}'.format(f.__name__, trace))
                raise
        wrapped_func.__name__ = f.__name__
        return wrapped_func


class count_calls:
    '''This is a decorator for counting how many times a function is called.

    Parameters
    ----------
    max_calls : int, Optional
        - If specified, it will throw an error if the number of calls
          exceeds this number. If nothing is specified then it will not
          throw an error
    '''
    def __init__(self, max_calls=None):
        if max_calls is None:
            max_calls = np.inf
        self.max_calls = max_calls
        self.calls = 0
        # self.f = f

    def __call__(self, f):
        def wrapped_func(*args, **kwargs):
            self.calls += 1
            if self.calls > self.max_calls:
                raise ValueError('In `{}` - Max calls reached: {}'.format(
                    wrapped_func.__name__, self.max_calls))
            return f(*args,**kwargs)
        wrapped_func.__name__ = f.__name__
        return wrapped_func


class Timer:
    '''Simple timer class

    Example use:
        >>> import numpy as np
        >>> a = np.arange(10000000)
        >>> b = 0
        >>> with Timer():
        >>>     for i in a:
        >>>         b += i 
        time: 2.00000001225 seconds
    '''
    def __init__(self):
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = default_timer()
        return self

    def __exit__(self, *args):
        self.end = default_timer()
        print('time: {} seconds'.format(self.end - self.start)) 

