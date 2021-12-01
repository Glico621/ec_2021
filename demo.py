# ECコンペ2021 単目的部門の実行サンプル
# 作成者は後藤裕介(芝浦工業大学)です．お問い合わせは y-goto@shibaura-it.ac.jp までお願いします．

# ### 概要
# GAで近似解を導出するpythonプログラムです．
# 最終世代の最良解をcsvとして書き出します．
# シミュレーションプログラムに渡す引数の設定，戻り値の受取り方，とりあえず動く実装として参考になさってください．
#
# ### 実行方法と注意
# プログラムのあるディレクトリで以下のように実行してください．
# python example_sop.py
# このとき，必ず実行用の変数の設定をご自分の環境に合わせて確認してください．
# 1つの解の評価に MacBook Air (M1, 2020)の環境で20秒程度かかります．
# 1つの子プロセスの展開で実行時に150MB程度のメモリを消費しますので，子プロセスの展開数は
# メモリとCPUコア数とを確認されてから設定してください．
#
# ### 動作環境
# 以下の環境で動作確認をしています．
# 外部のライブラリとしては[DEAP](https://github.com/deap/deap)を使っています．
# - on macOS Big Sur 11.1
# - deap: 1.3.1
# - multiprocess: 0.70.12.2
# - numpy: 1.20.2
# - pandas: 1.2.4
# - python: 3.9.2
# subprocessの処理でエラーが出る際には，pythonのバージョンを3.7以上に上げることを試してみてください．
import platform
import random
import subprocess

import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import tools
from example_mop import SIM_PATH

### 実行用の変数の設定
# - N_PROC: 子プロセスの展開数．
# - OUT_DIR: パレートフロントのcsvを書き出すディレクトリ．当初の設定ではカレントディレクトリを指定しています．
# - EID: パレートフロントのcsvの拡張子の前の部分． "p001" とすると，p001.csv として保存します．
# - FID: 目的関数のID．F_1では "[1]" F_2では "[2]" のように指定してください．
# - CITY: 実行する都市名． naha：沖縄県那覇市，hakodate: 北海道函館市．　
# - SEEDS: 実行時の乱数シードのリスト．""で囲って定義してください．
N_PROC = 5
OUT_DIR = "./"
EID = "p002"
FID = "[2]"
CITY = "hakodate"
# 単目的部門では， FID "[1]" CITY "naha"　， FID "[2]" CITY "hakodate" の2通りの指定を行えばよいです．
#!これが本番わからんらしい
SEEDS = "[123,42,256]"

### GAの設定
# - SEED：GAの遺伝的操作の際の乱数シード．シミュレーションにわたす乱数シードとは異なる点に注意．
# - N_IND：個体数
# - N_GEN：世代数
# - N_ATTR：支給対象を決める部分の遺伝子長．コーディングのしかたによって変更はありえます．
# - N_PAY: 支給金額を決める部分の遺伝子長．例えば，給付金額の調整を細かく行う際には変更が必要．
# - S_TOUR: トーナメントサイズ
# - P_CROSS_1：交叉確率（交叉を行うかどうか決定する確率）
# - P_CROSS_2：交叉確率（一様交叉を行うときに，その遺伝子座が交叉する確率）
# - P_MUTATION：各遺伝子座が突然変異する確率
# - N_HOF: 記録用に保持する(上位n個の)最良個体数
SEED = 42
N_IND = 5
N_GEN = 5
N_ATTR = 47
N_PAY = 16
S_TOUR = 3
P_CROSS_1 = 0.5
P_CROSS_2 = 0.5
P_MUTATION = 0.025
N_HOF = 20

# シミュレータのパス
SIM_PATH = platform.system() + "/syn_pop.py"


## 関数群の定義
# - gene2pay: コーディングした遺伝子から，設計変数へと変換する
# - ret_fitness: 子プロセスが完了することを待って，適応度を返す
# - evaluation: 個体の評価を行う．
# - decode_hof: 最良個体を支援制度（クエリ， 金額）にデコードする支援制度（クエリ， 金額）にデコードする
# - create_valid_pop: 支給対象の制約条件を満たす初期個体を生成する
def gene2pay(gene): 
    ### コーディングした遺伝子から，設計変数へと変換する関数
    # クエリ q は pandas.DataFrame.query の形式で書く形です．
    # シミュレーションプログラムでは制約条件を満たしているかの判定を渡されたクエリの文字列から
    # 行っていますので，スペースの入れ方をここでなされているように書いてください．
    #
    # 引数：
    #   gene: 個体の遺伝子
    # 戻り値：
    #   q: 給付金の対象を決めるクエリ
    #   pay: 給付金額（単位：万円）
    q = ''
    
    family_type_val = [0, 1, 2, 3, 4, 50, 60, 70, 80]
                                                #!range(0, 9) : 0,1,2,3,4,5,6,7,8,9
    family_type = [family_type_val[j] for i,j in zip(range(0, 9), range(9)) if gene[i] == 1]
        #! map() : 引数の関数を適用した後，結果のマップオブジェクトを返す
    family_type = ",".join(map(str, family_type))
    q = q + 'family_type_id == [' + family_type + ']'
    #! 実行結果  'family_type_id == [0,1,2,3,4,50,60,70,80]'


    role_household_type_val = [0, 1, 10, 11, 20, 21, 30, 31]
    role_household_type = [role_household_type_val[j] for i,j in zip(range(9, 17), range(8)) if gene[i] == 1]
    role_household_type = ",".join(map(str, role_household_type))
    q = q + ' and role_household_type_id == [' + role_household_type + ']'
    #! ここまで  'family_type_id == [0,1,2,3,4,50,60,70,80] and role_household_type_id == [0,1,10,11,20,21,30,31]'


    industry_type_val = [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    industry_type = [industry_type_val[j] for i,j in zip(range(17, 38), range(21)) if gene[i] == 1]
    industry_type = ",".join(map(str, industry_type))
    q = q + ' and industry_type_id == [' + industry_type + ']'
    #! ここまで  'family_type_id == [0,1,2,3,4,50,60,70,80] and role_household_type_id == [0,1,10,11,20,21,30,31] and industry_type_id == [-1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]'


    employment_type_val = [-1, 10, 20 ,30]
    employment_type = [employment_type_val[j] for i,j in zip(range(38, 42), range(4)) if gene[i] == 1]
    employment_type = ",".join(map(str, employment_type))
    q = q + ' and employment_type_id == [' + employment_type + ']'
    #! ここまで  'family_type_id == [0,1,2,3,4,50,60,70,80] and role_household_type_id == [0,1,10,11,20,21,30,31] and industry_type_id == [-1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] and employment_type_id == [-1,10,20,30]'


    company_size_val = [-1, 5, 10 ,100, 1000]
    company_size = [company_size_val[j] for i,j in zip(range(42, 47), range(5)) if gene[i] == 1]
    company_size = ",".join(map(str, company_size))
    q = q + ' and company_size_id == [' + company_size + ']'
    #! ここまで  'family_type_id == [0,1,2,3,4,50,60,70,80] and role_household_type_id == [0,1,10,11,20,21,30,31] and industry_type_id == [-1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] and employment_type_id == [-1,10,20,30] and company_size_id == [-1,5,10,100,1000]'


    pay = 0
    for i in range(47, 47 + N_PAY):
        pay += gene[i]

    return q, pay



def ret_fitness(p):
    ### 子プロセスが完了することを待って，目的関数値などを返す関数
    # 引数：
    #   p: 子プロセス
    # 戻り値：
    #   Fの目的関数値，Fの各条件での目的関数値,解が制約条件を満たすか（T/F），
    #   解の金額面の余裕（マイナスの場合には制約を満たしていない）
    
    a, err = p.communicate(timeout=1_000)
    # 正常に子プロセスが終了しないときは，目的関数値を1_000にしておく -> 次は選ばれないように
    if p.returncode != 0:
        print("sim failed %d %s %s" % (p.returncode, a, err))
            #!1_000 の下線は，読みやすくするためだけの下線 ,1000として解析される
        return 1_000, [1_000], False, [0]
    else:
        a_split = eval(a)
        # 実行
        if a_split[0] == None:
            return 1_000, a_split[1], a_split[2], a_split[3]
        else:
            return float(a_split[0]), a_split[1], a_split[2], a_split[3]


def evaluation(pop):
    ### 個体の評価を行う関数
    # 1個体の評価に時間がかかるため，並行して実行しています．
    #
    # 引数：
    #   pop: 個体の集合
    # 戻り値：
    #   pop: 評価値を計算した個体の集合
    
    # 各個体の評価値と実行可能かどうかをリストに入れていく
    f_list = []
    is_feasible_list = []

    # 1回あたりの実行に時間がかかるため，子プロセスを生成して，並行して実行する
    # 個体群をN_PROC個を単位として，バッチに分ける．
    # batch_list：バッチを要素とするlist
    # ind_list: 1バッチを構成する個体のlist
    n_ind = len(pop)
    batch_list, ind_list = [], []
    for i in range(n_ind): #0～n_ind-1まで
        ind_list.append(i)
        # 以下の条件でバッチにまとめる
        # (1)バッチで処理する子プロセスが満たされたとき
        # (2)(1)でないが，最後の個体となったとき
        if (i + 1) % N_PROC == 0 or i == n_ind - 1:
            batch_list.append(ind_list)
            ind_list = []

    # バッチごとに処理を進めていく
    # job_list: 実行するコマンドを要素とするlist
    # procs：subprocessに展開するためのlist
    for ind_list in batch_list:
        job_list, procs = [], []
        for i in ind_list:
            ind = pop[i]
            q, pay = gene2pay(ind)
            cmd = ["python", SIM_PATH, str(q), str(pay), str(FID), str(CITY), str(SEEDS)]
            job_list.append(cmd)
        #! subprocess：pythonからコマンドを実行するためのモジュール
        #! Popen：プログラムの起動
        procs = [subprocess.Popen(job, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for job in job_list]

        for i in range(len(ind_list)):
            # avg: 目的関数値
            # vals: 各条件での実行値のlist(valsを平均したものがavg)
            # judge: 解が制約条件（条件の優先関係）を満たしているか？
            # slacks: 金額面の制約の違反量（正の場合にはまだ余裕がある．負の場合には違反している量） 
            # vals, slacksはここでは使っていませんが，アルゴリズムやパラメータの検討時に参考になると思われます
            avg, vals, judge, slacks = ret_fitness(procs[i])
            f_list = f_list + [avg]
            is_feasible_list = is_feasible_list + [judge]

    for ind, f, j in zip(pop, f_list, is_feasible_list):
        # 目的関数値を各個体に割り当てていく．
        # このときに，解が金額の制約以外で，実行可能でないときには，ペナルティとして，目的関数値を0.0とする
        if j == False:
            ind.fitness.values = 1_000,
        else:
            ind.fitness.values = f,

    return pop


def decode_hof(hof):
    # パレートフロントの個体を支援制度（クエリ， 金額）にデコードする
    # 引数
    #   hof: パレートフロントの個体
    # 戻り値：
    #   支援制度（クエリ， 金額）のDataFrame
    q_and_pay = []
    for h in hof:
        q, p = gene2pay(h)
        f = h.fitness.values[0]
        q_and_pay.append([q, p, f])
    return pd.DataFrame(q_and_pay, columns=['query', 'payment', 'f'])

def create_valid_pop():
    ### 支給対象の定義において制約を満たす個体(群)を返す
    # 戻り値：
    #   個体群のリスト（2次元）
    
    true_list = [0,9,10,17,38,40,42,43] # これは必ず1を立てる
    order_list = [
        [
            [0],
            [4],
            [3],
            [1,2,5,6,7,8]
        ],
        [
            [9,10],
            [11,12,13,14,15,16]
        ],
        [
            [17], 
            [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        ],
        [
            [38],
            [40,41],
            [39]
        ],
        [
            [42],[43],[44],[45],[46]
        ],
        [
            [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
        ]
    ]#決める順番
    valid_pop = []
    for i in range(N_IND):
        tmp = [None for i in range(N_ATTR+N_PAY)]
        for attribute_type in order_list:
            constrations = True
            for priority_group in attribute_type:
                next_constrations = True
                for index in priority_group:
                    if index in true_list:
                        tmp[index] = 1
                    elif constrations and random.random() < 0.5:
                        tmp[index] = 1
                    else:
                        tmp[index] = 0
                        next_constrations = False
                constrations = next_constrations
        valid_pop.append(tmp)
    return valid_pop

def main():
    ### メインルーチン
    # GAはDEAPを使って実装する
    # 詳細は https://deap.readthedocs.io/en/master/index.html
    # 遺伝子：0 or 1で生成（ランダムに生成．生成/割当のしかたは改善の余地あり）
    # 交叉：一様交叉
    # 突然変異：ビット反転
    # 選択：トーナメント選択
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    # 初期の個体は支給対象の定義においては制約を満たす個体で始める（なお，金額面は満たすとは限らない）
    random.seed(SEED)
    valid_pop = create_valid_pop()

    def initPopulation(pcls, ind_init, file):
        return pcls(ind_init(c) for c in file)

    toolbox.register("population_byhand", initPopulation, list, creator.Individual, valid_pop)
    toolbox.register("mate", tools.cxUniform)
    toolbox.register("mutate", tools.mutFlipBit, indpb=P_MUTATION)
    toolbox.register("select", tools.selTournament, tournsize=S_TOUR)
    
    # 世代数の1/5ごとと，最後の世代において，以下のアーカイブ（スナップショット）を記録しておく
    # - pop_archive: 個体情報のアーカイブ
    # - paretof_archive: パレートフロントのアーカイブ
    pop_archive = []
    paretof_archive = []
    
    # 個体集合の作成
    pop = toolbox.population_byhand()
    pop_archive.append((0, pop[:]))
    # 個体の評価
    pop = evaluation(pop)
    
    # ログ関係
    stats = tools.Statistics()
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "avg", "std", "min", "max"
    record = stats.compile([ind.fitness.values[0] for ind in pop])
    logbook.record(gen=0, evals=len(pop), **record)
    hof = tools.HallOfFame(maxsize=N_HOF)
    
    # 進化のサイクルを回す
    for g in range(1, N_GEN + 1):
        # 子の世代の選択と複製
        
        print(g)
        
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSS_1:
                toolbox.mate(child1, child2, P_CROSS_2)
                del child1.fitness.values
                del child2.fitness.values
        # 突然変異
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # 子の世代で無効な適応度（delされたもの）をもつ個体を対象として評価を行う
            #!リスト内包
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        invalid_ind = evaluation(invalid_ind)
        # 子の世代を次の個体集合へ置き換える
        pop[:] = offspring

        record = stats.compile([ind.fitness.values for ind in pop])
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        hof.update(pop)
        
    
    # 次回の実行のため，削除しておく
    del creator.FitnessMin
    del creator.Individual
    
    return logbook, hof

if __name__ == "__main__":
    # 進化計算の実行
    logbook, hof = main() # logbookはこのサンプルでは利用していない
    
    # 最良個体の出力
    df_hof_final = decode_hof(hof)
    df_hof_final.drop_duplicates(keep='first', subset=['query', 'payment'])
    df_hof_final.to_csv(OUT_DIR + EID + '_p.csv')
