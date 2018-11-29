"""
    ===============================================
    :mod:`ec` -- Evolutionary computation framework
    ===============================================
    
    This module provides a framework for creating evolutionary computations.
    
    .. Copyright 2012 Aaron Garrett

    .. Permission is hereby granted, free of charge, to any person obtaining a copy
       of this software and associated documentation files (the "Software"), to deal
       in the Software without restriction, including without limitation the rights
       to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
       copies of the Software, and to permit persons to whom the Software is
       furnished to do so, subject to the following conditions:

    .. The above copyright notice and this permission notice shall be included in
       all copies or substantial portions of the Software.

    .. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
       IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
       FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
       AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
       LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
       OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
       THE SOFTWARE.       
        
    .. moduleauthor:: Aaron Garrett <garrett@inspiredintelligence.io>
"""
from inspyred_numpy.ec.ec import Bounder
from inspyred_numpy.ec.ec import DEA
from inspyred_numpy.ec.ec import DiscreteBounder
from inspyred_numpy.ec.ec import EDA
from inspyred_numpy.ec.ec import Error
from inspyred_numpy.ec.ec import ES
from inspyred_numpy.ec.ec import EvolutionaryComputation
from inspyred_numpy.ec.ec import EvolutionExit
from inspyred_numpy.ec.ec import GA
from inspyred_numpy.ec.ec import Individual
from inspyred_numpy.ec.ec import SA
from inspyred_numpy.ec import analysis
from inspyred_numpy.ec import archivers
from inspyred_numpy.ec import emo
from inspyred_numpy.ec import evaluators
from inspyred_numpy.ec import migrators
from inspyred_numpy.ec import observers
from inspyred_numpy.ec import replacers
from inspyred_numpy.ec import selectors
from inspyred_numpy.ec import terminators
from inspyred_numpy.ec import utilities
from inspyred_numpy.ec import variators

__all__ = ['Bounder', 'DiscreteBounder', 'Individual', 'Error', 'EvolutionExit', 
           'EvolutionaryComputation', 'GA', 'ES', 'EDA', 'DEA', 'SA',
           'analysis', 'archivers', 'emo', 'evaluators', 'migrators', 'observers', 
           'replacers', 'selectors', 'terminators', 'utilities', 'variators']


