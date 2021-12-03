#WOAクラスの定義

#getArray()とgetScore()を探す あとproblem


import math
import random

import numpy as np

class   WOA():
    def __init__(self,
                whale_max=10,              #クジラの頭数
                a_decrease=0.001,       #変数aの減少値
                logarithmic_spiral=1,   #対数螺旋の係数
    ):
        #各変数の初期化
        self.whale_max = whale_max
        self.a_decrease = a_decrease
        self.logarithmic_spiral = logarithmic_spiral


    def init(self, problem):
        self.problem = problem

        self.best_whale = None
        self.whales = []

        for _ in range(self.whale_max):
            o = problem.create()
            self.whales.append(o)

            if self.best_whale is None or self.best_whale.getScore() < o.getScore():
                self.best_whale = o.copy()

        self._a = 2


    def step(self):

        for whale in self.whales:
            pos = np.asarray(whale.getArray())

            if random.random() < 0.5:
                r1 = np.random.rand(self.problem.size)      #01乱数
                r2 = np.random.rand(self.problem.size)

                A = (2.0 * np.multiply(self._a, r1)) - self._a
                C = 2.0 * r2

                if np.linalg.norm(A) < 1:   #np.linalg.norm():行列ノルムを計算
                    #獲物に近づく
                    new_pos = self.best_whale.getArray()
                else:
                    #獲物を探す
                    new_pos = self.whales[random.randint(0, len(self.whales)-1)].getArray()

                new_pos = np.asarray(new_pos)

                D = np.linalg.norm(best_pos - pos)
                pos = new_pos - np.multiply(A, D)

            else:
                #旋回
                best_pos = np.asarray(self.best_whale.getArray())

                D = np.linalg.norm(best_pos - pos)
                L = np.random.uniform(-1, 1, self.problem.size)     #[-1, 1]乱数

                _b = self.logarithmic_spiral
                pos = np.multiply(np.multiply(D, np.exp(_b*L))), np.cos(2.0*np.pi*L) + best_pos

            whale.setArray(pos)
            if self.best_whale.getScore() < whale.getScore():
                self.best_whale = whale.copy()

        print(self.best_whale)
        #毎処理の最後に，aを減少させる
        self._a -= self.a_decrease
        if self._a < 0:
            self._a = 0

WOA(10, a_decrease=2/50, logarithmic_spiral=1)