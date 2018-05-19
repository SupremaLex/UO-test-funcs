from .CUTE import *
from .functions import *
from .diagonal_functions import *
from .quadratic_functions import *
from .full_hessian_functions import *
from .DIXMAAN import DIXMAAN

test_function_register = dict()
test_function_register["ExtendedFreudenstein&Roth"] = ext_freudenstein_and_roth
test_function_register["ExtendedTrigonometric"] = ext_trigonometric
test_function_register["ExtendedRosenbrock"] = ext_rosenbrock
test_function_register["GeneralizedRosenbrock"] = gen_rosenbrock
test_function_register["ExtendedWhiteHolst"] = ext_white_and_holst
test_function_register["ExtendedBeale"] = ext_beale
test_function_register["ExtendedPenalty"] = ext_penalty
test_function_register["Raydan1"] = raydan_1
test_function_register["Raydan2"] = raydan_2
test_function_register["Hager"] = hager
test_function_register["ExtendedThreeExponentialTerms"] = ext_three_exponential_terms
test_function_register["ExtendedHimmelblau"] = ext_himmelblau
test_function_register["GeneralizedWhiteHolst"] = gen_white_and_holst
test_function_register["GeneralizedPSC1"] = gen_PCS1
test_function_register["ExtendedPSC1"] = ext_PSC1
test_function_register["ExtendedPowell"] = ext_powell
test_function_register["ExtendedMaratos"] = ext_maratos
test_function_register["ExtendedCliff"] = ext_cliff
test_function_register["ExtendedWood"] = ext_wood
test_function_register["ExtendedHiebert"] = ext_hiebert
test_function_register["Staircase1"] = staircase_1
test_function_register["Staircase2"] = staircase_2
test_function_register["Diagonal1"] = diagonal_1
test_function_register["Diagonal2"] = diagonal_2
test_function_register["Diagonal3"] = diagonal_3
test_function_register["Diagonal4"] = diagonal_4
test_function_register["Diagonal5"] = diagonal_5
test_function_register["Diagonal6"] = diagonal_6
test_function_register["Diagonal7"] = diagonal_7
test_function_register["Diagonal8"] = diagonal_8
test_function_register["Diagonal9"] = diagonal_9
test_function_register["ExtendedBlockDiagonal"] = ext_block_diagonal
test_function_register["ExtendedTridiagonal1"] = ext_tridiagonal_1
test_function_register["ExtendedTridiagonal2"] = ext_tridiagonal_2
test_function_register["GeneralizedTridiagonal1"] = gen_tridiagonal_1
test_function_register["GeneralizedTridiagonal2"] = gen_tridiagonal_2
test_function_register["BroydenTridiagonal"] = broyden_tridiagonal
test_function_register["PerturbedTridiagonalQuadratic"] = perturbed_tridiagonal_quadratic
test_function_register["PerturbedQuadratic"] = perturbed_quadratic
test_function_register["QuadraticQF1"] = quadratic_1
test_function_register["QuadraticQF2"] = quadratic_2
test_function_register["ExtendedQuadraticPenalty1"] = ext_quadratic_penalty_1
test_function_register["ExtendedQuadraticPenalty2"] = ext_quadratic_penalty_2
test_function_register["ExtendedQuadraticExponential"] = ext_quadratic_exponential
test_function_register["PartialPerturbedQuadratic"] = partial_perturbed_quadratic
test_function_register["AlmostPerturbedQuadratic"] = almost_perturbed_quadratic
test_function_register["Generalized Quadratic"] = gen_quadratic
test_function_register["FullHessian1"] = full_hessian_1
test_function_register["FullHessian2"] = full_hessian_2
test_function_register["FullHessian3"] = full_hessian_3
test_function_register["FLETCBV3"] = FLETCBV3
test_function_register["FLETCHCR"] = FLETCHCR
test_function_register["BDQRTIC"] = BDQRTIC
test_function_register["TRIDIA"] = TRIDIA
test_function_register["ARGLINB"] = ARGLINB
test_function_register["ARWHEAD"] = ARWHEAD
test_function_register["NONDIA"] = NONDIA
test_function_register["NONDQUAR"] = NONDQUAR
test_function_register["DQDRTIC"] = DQDRTIC
test_function_register["EG2"] = EG2
test_function_register["CURLY20"] = CURLY20
test_function_register["DIXMAANA"] = DIXMAAN('A')
test_function_register["DIXMAANB"] = DIXMAAN('B')
test_function_register["DIXMAANC"] = DIXMAAN('C')
test_function_register["DIXMAAND"] = DIXMAAN('D')
test_function_register["DIXMAANE"] = DIXMAAN('E')
test_function_register["DIXMAANF"] = DIXMAAN('F')
test_function_register["DIXMAANG"] = DIXMAAN('G')
test_function_register["DIXMAANH"] = DIXMAAN('H')
test_function_register["DIXMAANI"] = DIXMAAN('I')
test_function_register["DIXMAANJ"] = DIXMAAN('J')
test_function_register["DIXMAANK"] = DIXMAAN('K')
test_function_register["DIXMAANL"] = DIXMAAN('L')
test_function_register["LIARWHD1"] = LIARWHD_1
test_function_register["POWER"] = POWER
test_function_register["ENGVAL1"] = ENGVAL1
test_function_register["CRAGGLVY"] = CRAGGLVY
test_function_register["EDENSCH"] = EDENSCH
test_function_register["INDEF"] = INDEF
test_function_register["CUBE"] = CUBE
test_function_register["EXPLIN1"] = EXPLIN1
test_function_register["EXPLIN2"] = EXPLIN2
test_function_register["ARGLINC"] = ARGLINC
test_function_register["BDEXP"] = BDEXP
test_function_register["HARKERP2"] = HARKERP2
test_function_register["GENHUMPS"] = GENHUMPS
test_function_register["MCCORMCK"] = MCCORMCK
test_function_register["NONSCOMP"] = NONSCOMP
test_function_register["VARDIM"] = VARDIM
test_function_register["QUARTC"] = QUARTC
test_function_register["SINQUAD"] = SINQUAD
test_function_register["ExtendedDENSCHNB"] = ext_DENSCHNB
test_function_register["ExtendedDENSCHNF"] = ext_DENSCHNF
test_function_register["LIARWHD2"] = LIARWHD_2
test_function_register["DIXON3DQ"] = DIXON3DQ
test_function_register["COSINE"] = COSINE
test_function_register["SINE"] = SINE
test_function_register["BIGGSB1"] = BIGGSB1
test_function_register["SINCOS"] = SINCOS
test_function_register["HIMMELBG"] = HIMMELBG
test_function_register["HIMMELH"] = HIMMELH