### ECコンペ2021 多目的部門の実行サンプルコード(example_mop.py)
# 作成者は後藤裕介(芝浦工業大学)です．お問い合わせは y-goto@shibaura-it.ac.jp までお願いします．
# 
# ■■■ 概要 ■■■
# NSGA-IIで近似解を導出するpythonプログラムです．
# 最終世代のパレートフロントをcsvとして書き出します．
# シミュレーションプログラムに渡す引数の設定，戻り値の受取り方，とりあえず動く実装として参考になさってください．
# 
# ■■■ 実行方法と注意 ■■■
# プログラムのあるディレクトリで以下のように実行してください．
# python example_mop.py
# このとき，必ず実行用の変数の設定をご自分の環境に合わせて確認してください．
# 1つの解の評価に MacBook Air (M1, 2020)の環境で40秒程度かかります．
# 1つの子プロセスの展開で実行時に150MB程度のメモリを消費しますので，子プロセスの展開数はメモリとCPUコア数とを
# 確認されてから設定してください．
# 
# ■■■ 動作環境 ■■■
# 以下の環境で動作確認をしています．
# 外部のライブラリとしてはDEAPを使っています．
# https://github.com/deap/deap
# 
# 動作の確認を行っている環境(on macOS Big Sur 11.1)：
# - deap: 1.3.1
# - multiprocess: 0.70.12.2
# - numpy: 1.20.2
# - pandas: 1.2.4
# - python: 3.9.2
# subprocessの処理でエラーが出る際には，pythonのバージョンを3.7以上に上げることを試してみてください．
import platform
import random
import subprocess

import pandas as pd
from deap import base
from deap import creator
from deap import tools
from deap.benchmarks.tools import hypervolume

### 実行用の変数の設定
# N_PROC: 子プロセスの展開数．
# OUT_DIR: パレートフロントのcsvを書き出すディレクトリ．当初の設定ではカレントディレクトリを指定しています．
# EID: パレートフロントのcsvの拡張子の前の部分． "p001" とすると，p001.csv として保存します．
# CITY: 実行する都市名． naha：沖縄県那覇市，hakodate: 北海道函館市
# SEEDS: 実行時の乱数シードのリスト．""で囲って定義してください．
N_PROC = 5
OUT_DIR = "./"
EID = "p001"
CITY = "naha"
SEEDS = "[123,42,256]"

### GAの設定
# SEED：GAの遺伝的操作の際の乱数シード．シミュレーションにわたす乱数シードとは異なる点に注意．
# N_IND：個体数
# N_GEN：世代数
# N_ATTR：支給対象を決める部分の遺伝子長．コーディングのしかたによって変更はありえます．
# N_PAY: 支給金額を決める部分の遺伝子長．例えば，給付金額の調整を細かく行う際には変更が必要．
# P_CROSS_1：交叉確率（交叉を行うかどうか決定する確率）
# P_CROSS_2：交叉確率（一様交叉を行うときに，その遺伝子座が交叉する確率）
# P_MUTATION：各遺伝子座が突然変異する確率
SEED = 42
N_IND = 40
N_GEN = 2
N_ATTR = 47
N_PAY = 16
P_CROSS_1 = 0.9
P_CROSS_2 = 0.8
P_MUTATION = 0.025

### Hypervolume計算用
# REF_P: 参照点の座標
REF_P = [-0.1, -0.1]

# シミュレータのパス
SIM_PATH = platform.system() + "/syn_pop.py"


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
    family_type = [family_type_val[j] for i,j in zip(range(0, 9), range(9)) if gene[i] == 1]
    family_type = ",".join(map(str, family_type))
    q = q + 'family_type_id == [' + family_type + ']'

    role_household_type_val = [0, 1, 10, 11, 20, 21, 30, 31]
    role_household_type = [role_household_type_val[j] for i,j in zip(range(9, 17), range(8)) if gene[i] == 1]
    role_household_type = ",".join(map(str, role_household_type))
    q = q + ' and role_household_type_id == [' + role_household_type + ']'

    industry_type_val = [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    industry_type = [industry_type_val[j] for i,j in zip(range(17, 38), range(21)) if gene[i] == 1]
    industry_type = ",".join(map(str, industry_type))
    q = q + ' and industry_type_id == [' + industry_type + ']'

    employment_type_val = [-1, 10, 20 ,30]
    employment_type = [employment_type_val[j] for i,j in zip(range(38, 42), range(4)) if gene[i] == 1]
    employment_type = ",".join(map(str, employment_type))
    q = q + ' and employment_type_id == [' + employment_type + ']'        

    company_size_val = [-1, 5, 10 ,100, 1000]
    company_size = [company_size_val[j] for i,j in zip(range(42, 47), range(5)) if gene[i] == 1]
    company_size = ",".join(map(str, company_size))
    q = q + ' and company_size_id == [' + company_size + ']'

    pay = 0
    for i in range(47, 47 + N_PAY):
        pay += gene[i]

    return q, pay

def ret_fitness(p):
    ### 子プロセスが完了することを待って，目的関数値などを返す関数
    # 引数：
    #   p: 子プロセス
    # 戻り値：
    #   F1の目的関数値，F1の各条件での目的関数値，F2の目的関数値，F2の各条件での目的関数値，解が制約条件を満たすか（T/F），
    #   解の金額面の余裕（マイナスの場合には制約を満たしていない）
    
    a, err = p.communicate(timeout=1_000)
    # 正常に子プロセスが終了しないときは，目的関数値を1_000にしておく -> 次は選ばれないように
    if p.returncode != 0:
        print("sim failed %d %s %s" % (p.returncode, a, err))
        return 1_000, [1_000], 1_000, [1_000], False, [0]
    else:
        a_split = eval(a)
        if a_split[0] == None or a_split[2] == None:
            return 1_000, a_split[1], 1_000, a_split[3], a_split[4], a_split[5]
        else:
            return float(a_split[0]), a_split[1], float(a_split[2]), a_split[3], a_split[4], a_split[5]

def evaluation(pop):
    ### 個体の評価を行う関数
    # 1個体の評価に時間がかかるため，並行して実行しています．
    # 
    # 引数：
    #   pop: 個体の集合
    # 戻り値：
    #   pop: 評価値を計算した個体の集合
    
    # 各個体の評価値と実行可能かどうかをリストに入れていく
    f1_list, f2_list = [], []
    is_feasible_list = []

    # 1回あたりの実行に時間がかかるため，子プロセスを生成して，並行して実行する
    # 個体群をN_PROC個を単位として，バッチに分ける．
    # batch_list：バッチを要素とするlist
    # ind_list: 1バッチを構成する個体のlist
    n_ind = len(pop)
    batch_list, ind_list = [], []
    for i in range(n_ind):
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
            cmd = ["python", SIM_PATH, str(q), str(pay), "[1,2]", str(CITY), str(SEEDS)]
            job_list.append(cmd)
        procs = [subprocess.Popen(job, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for job in job_list]

        for i in range(len(ind_list)):
            # avg: 目的関数値
            # vals: 各条件での実行値のlist(valsを平均したものがavg)
            # feasible: 解が制約条件（条件の優先関係）を満たしているか？
            # slacks: 金額面の制約の違反量（正の場合にはまだ余裕がある．負の場合には違反している量） 
            # vals, slacksはここでは使っていませんが，アルゴリズムやパラメータの検討時に参考になると思われます
            avg_1, vals_1, avg_2, vals_2, feasible, slacks = ret_fitness(procs[i])
            f1_list = f1_list + [avg_1]
            f2_list = f2_list + [avg_2]
            is_feasible_list = is_feasible_list + [feasible]

    for ind, f1, f2, j in zip(pop, f1_list, f2_list, is_feasible_list):
        # 目的関数値を各個体に割り当てていく．
        # このときに，解が金額の制約以外で，実行可能でないときには，ペナルティとして，目的関数値を1_000とする
        if j == False:
            ind.fitness.values = 1_000, 1_000,
        else:
            ind.fitness.values = f1, f2,
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
        f1 = h.fitness.values[0]
        f2 = h.fitness.values[1]
        q_and_pay.append([q, p, f1, f2])
    return pd.DataFrame(q_and_pay, columns=['query', 'payment', 'f_1', 'f_2']) 

def create_valid_pop():
    ### 支給対象の定義において制約を満たす個体(群)を返す
    # 戻り値：
    #   個体群のリスト（2次元）
    true_list = [0,9,10,17,38,40,42,43] # これは必ず1を立てる
    valid_pop = []
    for i in range(N_IND):
        tmp = []
        # - 最低の支給（単身で無職）範囲の確定
        # - それ以外の部分は 0.5 の確率で 1 を割当
        for j in range(N_ATTR+N_PAY):
            if j in true_list:
                tmp.append(1)
            elif random.random() < 0.5:
                tmp.append(1)
            else:
                tmp.append(0)
        valid_pop.append(tmp)
    return valid_pop

def main():
    ### メインルーチン
    # GAはDEAPを使って実装する
    # 詳細は https://deap.readthedocs.io/en/master/index.html
    # 遺伝子：0 or 1で生成（ランダムに生成．生成/割当のしかたは改善の余地あり）
    # 交叉：一様交叉
    # 突然変異：ビット反転
    # 選択：NSGA-II
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
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
    toolbox.register("select", tools.selNSGA2)
    
    # 個体集合の作成
    pop = toolbox.population_byhand()
    # 個体の評価
    pop = evaluation(pop)
    # この選択(select)は，実際には何もしないが（選択はしないが），NSGA-IIで必要となる
    # crowding distanceを求めるために行われている
    # ref: https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
    # ref: https://github.com/DEAP/deap/blob/master/deap/tools/emo.py
    pop = toolbox.select(pop, len(pop))
    # パレートフロント
    paretof = tools.ParetoFront()

    # 進化のサイクルを回す
    for g in range(1, N_GEN + 1):
        # 子の世代の選択と複製
        offspring = tools.selTournamentDCD(pop, len(pop))
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
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        invalid_ind = evaluation(invalid_ind)
        # 子の世代を次の個体集合へ置き換える
        pop = toolbox.select(pop + offspring, N_IND)
        # パレートフロントの更新
        paretof.update(pop)
        
    print("Final population hypervolume is %f" % hypervolume(pop, REF_P))
    
    # 次回の実行のため，削除しておく
    del creator.FitnessMin
    del creator.Individual
    
    return paretof                        

if __name__ == "__main__":    
    paretof = main() # 進化計算の実行
    
    # パレートフロントを出力
    df_hof = decode_hof(paretof)
    df_hof.drop_duplicates(keep='first', subset=['query', 'payment'])
    df_hof.to_csv(OUT_DIR + EID + '.csv')
