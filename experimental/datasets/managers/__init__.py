"""__init__.py file for experimental.datasets.managers package."""

from experimental.datasets.managers.cars_test import CarsTest, CarsUnivariateTest, CarsAdjustments
from experimental.datasets.managers.default_test import DefaultTest, DefaultAdjustments
from experimental.datasets.managers.law_test import LawTest, LawAdjustments, LawResponse
from experimental.datasets.managers.puzzles_test import PuzzlesTest, PuzzlesResponse
from experimental.datasets.managers.restaurants_test import RestaurantsTest, RestaurantsAdjustment
from experimental.datasets.managers.synthetic_test import SyntheticTest, SyntheticAdjustments2D, SyntheticAdjustments3D, \
    SyntheticResponse
from experimental.datasets.managers.test_manager import Fold, TestManager, ClassificationTest, RegressionTest, \
    AnalysisCallback, BoundsAnalysis, DistanceAnalysis
