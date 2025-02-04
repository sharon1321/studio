"""
    Copyright 2019 Samsung SDS

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


from brightics.function.regression.mlp_regression import mlp_regression_train, mlp_regression_predict
from brightics.common.datasets import load_iris
import unittest
import pandas as pd
import numpy as np


class MLPRegression(unittest.TestCase):
    
    def setUp(self):
        print("*** MLP Regression Train/Predict UnitTest Start ***")
        self.testdata = load_iris()

    def tearDown(self):
        print("*** MLP Regression Train/Predict UnitTest End ***")
    
    def test(self):
        mlp_train = mlp_regression_train(self.testdata, feature_cols=['sepal_length', 'sepal_width', 'petal_length'], label_col='petal_width', random_state=12345)
        mlp_model = mlp_train['model']['mlp_model']
        intercepts = mlp_model.intercepts_
        coefficients = mlp_model.coefs_
        self.assertEqual(round(intercepts[0][0], 15), -0.083212718456184)
        self.assertEqual(round(intercepts[0][1], 15), 0.144096440665653)
        self.assertEqual(round(intercepts[0][2], 15), -0.099062732757039)
        self.assertEqual(round(intercepts[0][97], 15), -0.163275298693195)
        self.assertEqual(round(intercepts[0][98], 15), -0.085856149507296)
        self.assertEqual(round(intercepts[0][99], 15), 0.065225516018867)
        self.assertEqual(round(intercepts[1][0], 15), -0.116788808184757)
        self.assertEqual(round(coefficients[0][0][0], 15), 0.223619772736289)
        self.assertEqual(round(coefficients[0][0][1], 15), -0.077509711374316)
        self.assertEqual(round(coefficients[0][0][2], 15), -0.137670061450746)
        self.assertEqual(round(coefficients[0][0][97], 15), 0.173790742388019)
        self.assertEqual(round(coefficients[0][0][98], 15), 0.175935880097173)
        self.assertEqual(round(coefficients[0][0][99], 15), -0.203271600229577)
        self.assertEqual(round(coefficients[0][1][0], 15), -0.132338401337374)
        self.assertEqual(round(coefficients[0][1][1], 15), -0.087304986819381)
        self.assertEqual(round(coefficients[0][1][2], 15), 0.052531063656606)
        self.assertEqual(round(coefficients[0][1][97], 15), -0.006299109647417)
        self.assertEqual(round(coefficients[0][1][98], 15), -0.135369358740921)
        self.assertEqual(round(coefficients[0][1][99], 15), 0.027438115074856)
        self.assertEqual(round(coefficients[0][2][0], 15), 0.186304588892986)
        self.assertEqual(round(coefficients[0][2][1], 15), -0.050314004244571)
        self.assertEqual(round(coefficients[0][2][2], 15), -0.027125306471382)
        self.assertEqual(round(coefficients[0][2][97], 15), 0.022353758081494)
        self.assertEqual(round(coefficients[0][2][98], 15), -0.021928185656289)
        self.assertEqual(round(coefficients[0][2][99], 15), -0.136309422981933)
        self.assertEqual(round(coefficients[1][0][0], 15), 0.222253396385915)
        self.assertEqual(round(coefficients[1][1][0], 15), -0.182476491105317)
        self.assertEqual(round(coefficients[1][2][0], 15), -0.191986097802643)
        self.assertEqual(round(coefficients[1][97][0], 15), 0.255668810206623)
        self.assertEqual(round(coefficients[1][98][0], 15), -0.165303406243865)
        self.assertEqual(round(coefficients[1][99][0], 15), 0.214519092446376)
        
        predict = mlp_regression_predict(self.testdata, mlp_train['model'])['out_table']['prediction']
        np.testing.assert_array_equal([round(x, 15) for x in predict], [1.273561890166563, 1.212772361115921, 1.155871317243720, 1.135099461540369, 1.238760408181506, 1.358853230242947, 1.122473847494187, 1.248870891826169, 1.074244177211045, 1.217479064681680, 1.362117004573845, 1.190395892161007, 1.186334498225570, 1.031313191933936, 1.458545366246964, 1.425425401622483, 1.346502498638738, 1.273561890166563, 1.455251931278644, 1.264035482482514, 1.363123038528146, 1.268382123112851, 1.104044632889204, 1.279740242076718, 1.203532451275072, 1.240503222811670, 1.252688705946628, 1.305611466998822, 1.301496757842151, 1.167421094424049, 1.191676576409287, 1.361771482290490, 1.282240521078502, 1.368905366834476, 1.217479064681680, 1.231998769913468, 1.378436529347472, 1.217479064681680, 1.070710027935359, 1.277096039442249, 1.239697631271636, 1.073746374277681, 1.065953812706181, 1.248546989741549, 1.282177076222894, 1.186334498225570, 1.266755351646127, 1.131167128754861, 1.330872044086847, 1.244756182669498, 1.956602205434725, 1.810118713086944, 1.949228103734809, 1.520728503446111, 1.818751423076241, 1.635311954088759, 1.811953059704987, 1.330244316531568, 1.845613018832819, 1.468495220220644, 1.349887098965034, 1.663705913073248, 1.621729215828674, 1.745873342313451, 1.533494951505848, 1.858442069590472, 1.624356676179209, 1.616391671348893, 1.713678507182365, 1.543614140703085, 1.726966626206546, 1.676346866918860, 1.784833867950051, 1.740352197116896, 1.774410740754129, 1.832109890766889, 1.901736916278462, 1.908954852052211, 1.706825804193291, 1.531122423730590, 1.507225375380181, 1.497698205813856, 1.597337332216242, 1.746933423252684, 1.579263641786706, 1.732123707550066, 1.887552995222444, 1.729320259230268, 1.589654738918094, 1.531830925579554, 1.571749542181581, 1.742869502601255, 1.601313290715846, 1.346888040873308, 1.582218285863294, 1.620544727228657, 1.615258144559063, 1.731789971374415, 1.343141549186556, 1.600632497725758, 1.919078252645713, 1.702941202349265, 2.073669989441601, 1.866593343097523, 1.933159594716896, 2.243332725316149, 1.443665102645676, 2.146243920485366, 1.949549005339442, 2.145844398741088, 1.884526990748753, 1.851973666280821, 1.973577344898622, 1.661458451828760, 1.708459431843921, 1.879579392263594, 1.907584639990645, 2.318024440099995, 2.272858747788893, 1.712651533841612, 2.023667173411891, 1.647414330222105, 2.262812988134450, 1.795876158343161, 1.985193182003127, 2.115234833046119, 1.770874750328305, 1.768444457327506, 1.883069766203627, 2.087142572835510, 2.145670373773970, 2.336444621979062, 1.883069766203627, 1.818447273357216, 1.806485401735814, 2.222705369075057, 1.889620429291602, 1.891108216884540, 1.737921904116097, 1.992571073489085, 1.965625906701268, 1.966996118762834, 1.702941202349265, 2.018719574926733, 1.985193182003127, 1.926004821869712, 1.793358852858802, 1.882009685264394, 1.849755146583515, 1.741499290539689]
)
