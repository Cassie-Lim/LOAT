'''
解析物体位置的一些公用函数
'''
import nltk
from nltk.tree import Tree
import json
from tqdm import tqdm
import alfred_utils.gen.constants as constants
import re
import os

SPLIT_PATH = "alfred_data_small/splits/oct21.json"
OTHER_NAME_MAP = {'chair':"ArmChair", 'bathtub':'BathtubBasin','table':['CoffeeTable','DiningTable','SideTable'],'sink':'SinkBasin','stove':'StoveBurner','tv':'TVStand','closet':'Cabinet','counter':'CounterTop','soap':'SoapBar','phone':'CellPhone','salt':'SaltShaker','shaker':['SaltShaker','PepperShaker'],'lamp':['FloorLamp','DeskLamp'],'cell':'CellPhone','computer':'Laptop','coffeemaker':'CoffeeMachine','cupboard':'Cabinet','bottle':['Glassbottle','SoapBottle'],'refrigerator':'Fridge','disc':"CD",'keys':'KeyChain','shelving':"Shelf",'trashbin':'GarbageCan','garbagebin':'GarbageCan','remote':'RemoteControl','couch':'Sofa','cupboard':'Cabinet'}
PICK_VERB = ['pick','get','move','place','take','grab'] #表示拿起动作的动词
ALFRED2FILM_NAMES = {'Sink':'SinkBasin'}

OPENABLE_CLASS_LOWER = [obj.lower() for obj in constants.OPENABLE_CLASS_LIST]

def rewrite_sentence(lan_feature,lan_gan):
    '''
    将语言特征重写为句子
    lan_gan:语言的粒度，high or low
    '''
    if lan_gan == "high":
        lan_feature = [word.strip() for word in lan_feature]
        lan_feature = " ".join(lan_feature)
        # NOTE high 的功能还未经测试
    else:
        lan_feature_new = []
        for sub_instr in lan_feature:
            sub_instr = [word.strip() for word in sub_instr]
            sub_instr = " ".join(sub_instr)
            lan_feature_new.append(sub_instr)
        lan_feature = ". ".join(lan_feature_new)
    lan_feature = re.sub('<<stop>>','',lan_feature)
    lan_feature = re.sub(' \' ','\'',lan_feature)
    lan_feature = re.sub(r"\s+(?=[.,!?])", "", lan_feature) #把标点符号前的空格都去掉
    lan_feature = re.sub(r" <<goal>>",r".",lan_feature)
    lan_feature = re.sub(r" \.\.",'. ',lan_feature)
    lan_feature = re.sub(r"\.\.",'. ',lan_feature)
    return lan_feature

def rewrite_parse_result_sentence(parse_tree):
    '''
    将解析结果重写为句子，以便能和语言特征给出的句子匹配
    '''
    parse_sentence = " ".join(parse_tree.leaves())
    parse_sentence = re.sub("\.","",parse_sentence)
    parse_sentence = parse_sentence.strip()
    parse_sentence = re.sub(r"\s+(?=[.,!?])", "", parse_sentence) #把标点符号前的空格都去掉
    parse_sentence = re.sub(r" \'", "\'", parse_sentence)
    return parse_sentence


# ******************
# 关于句法解析树的一些函数
def get_parent_loc(productions,rule_id):
    '''
    寻找第relu_id的左边节点,所在的位置（生成它的那个规则，以及在那个规则的位置）
    NOTE 这样找最近的一个点，不太对，因为最近的那个，可能是前面树的子树的
    应该再加一个判断条件：前面最近的一个，还没有产生式的
    '''
    parent = productions[rule_id]._lhs
    lproduc = 0
    for idx in range(rule_id-1,-1,-1):
        production = productions[idx]
        for j, rhs in enumerate(production._rhs):
            if parent == rhs:
                if lproduc!=0:
                    lproduc -= 1
                else:
                    return idx,j
        if production._lhs == parent:
            lproduc += 1

def get_parse_loc(productions,production_id:int,loc_id:int,loc:list):
    '''
    给出在产生式里，某个短语在句法树里的位置
    production_id:那个短语所在的产生式，在productins里的位置
    loc_id: 那个短语在产生式里面的位置
    注意：由于产生式都是左边一个，右边多个，因此给位置的时候，只给要找的短语在右边list的位置
    '''
    # 使用递归来写
    loc.append(loc_id)
    production_i = productions[production_id]
    parent = production_i._lhs
    if str(parent)=='ROOT':
        loc = loc[::-1]
    else:
        # 寻找父节点所在的规则号
        i,j = get_parent_loc(productions,production_id)
        loc = get_parse_loc(productions,i,j,loc)
    return loc
        ## NOTE：如果这里不加return的话，还是会继续执行for循环。感觉代码也没有很简单，应该用for循环

def get_treeloc_from_producid(productions,produc_idx):
    '''
    给出productions中第produc_idx个规则生成的树,即该树的根节点为productions[produc_idx]的lhs
    '''
    i,j = get_parent_loc(productions,produc_idx)
    tree_loc = get_parse_loc(productions,i,j,[])
    return tree_loc
# *******************

# *************
# 与名词匹配相关的一些函数
def noun_map(noun:str):
    '''
    名词映射：将language feature里面的名词映射到FILM list里面的名词
    NOTE:传进来的词要确保都已经小写了,不然可能匹配不出来
    '''
    if noun in constants.OBJECTS_SINGULAR:
        noun_new = constants.OBJECTS[constants.OBJECTS_SINGULAR.index(noun)]
    elif noun in constants.OBJECTS_PLURAL:
        noun_new = constants.OBJECTS[constants.OBJECTS_PLURAL.index(noun)]
    elif noun in OTHER_NAME_MAP:
        noun_new = OTHER_NAME_MAP[noun]
    else:
        noun_new = None
    if noun_new is not None:
        if not isinstance(noun_new,list):
            if noun_new in ALFRED2FILM_NAMES:
                noun_new = ALFRED2FILM_NAMES[noun_new]
        else:
            for i,noun in enumerate(noun_new):
                if noun in ALFRED2FILM_NAMES:
                    noun_new[i] = ALFRED2FILM_NAMES[noun]
    return noun_new

def nounphrase_map(noun_list:list):
    '''
    将名词或名词短语映射到alfred的名词
    '''
    noun_list = [word.lower() for word in noun_list]
    noun_phrase = "".join(noun_list)
    noun_alfred = noun_map(noun_phrase)
    if noun_alfred is None:
        # 匹配单个词语
        for noun in noun_list:
            noun_alfred = noun_map(noun)
            if noun_alfred is not None:
                break
    return noun_alfred
# ***********


# ****************
# 与句法解析相关的一些函数
def get_noun_phrase(np_tree:Tree):
    '''
    从名词短句中提取名词短语。NOTE：这里的名词短句需要非常短，不包含其它成分
    (PP (IN inside) (NP (DT the) (JJ safe))):由于错误标记，导致识别不出来
    '''
    # NOTE 高度是层数，包括根节点和叶节点
    noun_pharse = []
    if np_tree.height() > 3:
        # 可能名词后面还有修饰语，提取第一个名词
        noun_pharse = np_tree[0].leaves()
    else:
        for subtree in np_tree.subtrees(lambda x: x.label()=='NN' or x.label()=='NNS'):
            noun_pharse += subtree.leaves()
    return noun_pharse

def get_pos_loc_pp(rpp_tree):
    '''
    从pp（副词短语）里面匹配位置词，和location
    '''
    loc_word = [] #有可能这个结构里面一直都没有NP这个结构
    pos_word = []
    for subtree in rpp_tree.subtrees():
        # 这里应该用子树，直接写为for subtree in rpp_tree：只会遍历两个最主要的分支
        if subtree.label() == 'IN':
            pos_word = subtree.leaves()
        if subtree.label() == 'NP':
            loc_word = get_noun_phrase(subtree)
            break
    return pos_word, loc_word

def get_pos_loc_vp(vp_tree):
    '''
    从vp(动词短语)中提取方位名词和位置词
    '''
    # 找到第一个pp,从中匹配
    childrens = [str(child) for child in vp_tree.productions()[0]._rhs]
    if ("VBG" in childrens or "VBN" in childrens)and "PP" in childrens:
        pp_index = childrens.index("PP")
        pp_tree = vp_tree[pp_index]
        pos_word, loc_word = get_pos_loc_pp(pp_tree)
    else:
        pos_word = []
        loc_word = []
        # with open(vp_log_file, "a") as f:
        #     f.write("\n")
        #     f.write(str(vp_tree))
        # 感觉没啥问题
    return pos_word, loc_word

def parse_pp(np_tree:Tree):
    '''
    从"NP -> NP PP"结构中解析物体位置关系
    np_tree: NP -> NP PP这条规则生成的树，整个大NP是根节点
    从右边的np里面提取名词短语，pp里面提取方位词和名词短语
    NOTE 可能只适用于短语非常短的形式，没有考虑右边的NP仍然包含NP PP……等情况
    '''
    productions = np_tree.productions()[0]
    rnp_tree = np_tree[0]
    target_word = get_noun_phrase(rnp_tree)
    # 寻找PP短语的位置
    rhs_string = [str(rhs) for rhs in productions._rhs]
    pp_index = rhs_string.index("PP")
    rpp_tree = np_tree[pp_index]
    # 从pp里面提取方位词和位置
    pos_word, loc_word = get_pos_loc_pp(rpp_tree)
    return target_word, pos_word, loc_word

def parse_sbar(np_tree:Tree,debug = True):
    '''
    解析定语从句修饰的名词短语
    '''
    productions = np_tree.productions()[0]
    rnp_tree = np_tree[0]
    target_word = get_noun_phrase(rnp_tree)
    # 寻找sbar短语的位置
    rhs_string = [str(rhs) for rhs in productions._rhs]
    sbar_index = rhs_string.index("SBAR")
    sbar_tree = np_tree[sbar_index]
    for subtree in sbar_tree: #NOTE 这里只在最上层匹配，不知道会不会报错
        if subtree.label() == 'S':
            subsentence_tree = subtree
            break
    subsentence_production = subsentence_tree.productions()[0]
    if str(subsentence_production) == 'S -> VP':
        # 不是倒装
        vp_tree = subsentence_tree[0]
        vp_childrens = [str(child) for child in vp_tree.productions()[0]._rhs]
        if 'NP' in vp_childrens:
            np_index = vp_childrens.index('NP')
            pos_word = []
            loc_word = get_noun_phrase(vp_tree[np_index])
        elif 'PP' in vp_childrens:
            pp_index = vp_childrens.index('PP')
            pos_word, loc_word = get_pos_loc_pp(vp_tree[pp_index])
        elif 'VP' in vp_childrens:
            # 从vp短语中提取
            vp_index = vp_childrens.index('VP')
            pos_word, loc_word = get_pos_loc_vp(vp_tree[vp_index])
        else:
            # 仅仅只是一般的从句,不匹配
            target_word, pos_word, loc_word = [],[],[]
    elif str(subsentence_production) =='S -> NP VP':
        # 倒装结构,可能没有方位名词,直接提取其中的名词即可
        pos_word = []
        for subtree in sbar_tree.subtrees():
            if subtree.label() == 'NP':
                loc_word = get_noun_phrase(subtree)
                break
        temp = target_word
        target_word = loc_word
        loc_word = temp
    else:
        # # 仅仅只是一般的从句,不匹配
        target_word, pos_word, loc_word = [],[],[]
    return target_word, pos_word, loc_word

def parse_vp(np_tree:Tree,debug = True):
    '''
    从动词短语里面匹配位置
    '''
    productions = np_tree.productions()[0]
    rnp_tree = np_tree[0]
    target_word = get_noun_phrase(rnp_tree)
    rhs_string = [str(rhs) for rhs in productions._rhs]
    vp_index = rhs_string.index("VP")
    vp_tree = np_tree[vp_index]
    pos_word, loc_word = get_pos_loc_vp(vp_tree)
    return target_word,pos_word, loc_word

def parse_nppp(vp_tree:Tree,debug= True):
    '''
    解析 vp -> vb np pp结构
    '''
    productions = vp_tree.productions()[0]
    rhs_string = [str(rhs) for rhs in productions._rhs]
    vb_index = rhs_string.index("VB")
    vb_word = vp_tree[vb_index].leaves()[0]
    if vb_word in PICK_VERB:
        rnp_index = rhs_string.index("NP")
        rnp_tree = vp_tree[rnp_index]
        target_word = get_noun_phrase(rnp_tree)
        
        pp_index = rhs_string.index("PP")
        pp_tree = vp_tree[pp_index]
        pos_word, loc_word = get_pos_loc_pp(pp_tree)
        # if pos_word[0] != 'from':
        if len(pos_word)!=0 and pos_word[0] != 'from':
            # print(pos_word[0])
            pos_word, loc_word = [],[]
    else:
        # print(vb_word)
        target_word, pos_word, loc_word = [],[],[]
    return target_word,pos_word, loc_word
# ************

# *******************
# 将句法分析解析后的结果再次处理为位置关系形式的函数

def parse_locs_norm(locs):
    '''
    将locs里面的名词再次解析为与alfred(FILM)对应的名词,并将一些多余的,错误的位置关系去掉
    loc:list(tuple) tuple是三元组,分别为target position_word 和 location word
    '''
    loc_info = {}
    for loc_tuple in locs:
        target = loc_tuple[0]
        pos_word = loc_tuple[1]
        loc_word = loc_tuple[2]
        if pos_word != "with":
            target_alfred = nounphrase_map(target)
            loc_alfred = nounphrase_map(loc_word)
            if target_alfred is None or loc_alfred is None:
                continue
            if not isinstance(loc_alfred,list):
                loc_alfred = [loc_alfred]
            if target_alfred is not None and loc_alfred is not None:
                if isinstance(target_alfred,list):
                    for target_temp in target_alfred:
                        loc_info[target_temp] = loc_alfred
                else:
                    loc_info[target_alfred] = loc_alfred
    return loc_info
# *******************

# ************
# 与上层调用相关的一些函数
def get_loc_info_parse_string(parse_tree_string):
    '''
    从句法分析结果（string格式）获得位置信息
    '''
    parse_tree = Tree.fromstring(parse_tree_string)
    productions = parse_tree.productions()
    locs = []
    for produc_id,produc in enumerate(productions):
        produc_str = str(produc)
        if produc_str == "NP -> NP PP" or produc_str=="NP -> NP , PP ," or produc_str == "NP -> NP , PP":
            tree_loc = get_treeloc_from_producid(productions,produc_id)
            np_tree = parse_tree[tree_loc]
            target, pos, loc = parse_pp(np_tree)
            locs.append((target, pos, loc))
        elif produc_str == 'NP -> NP SBAR':
            # 通过定语从句来匹配
            tree_loc = get_treeloc_from_producid(productions,produc_id)
            np_tree = parse_tree[tree_loc]
            target, pos, loc = parse_sbar(np_tree)
            locs.append((target, pos, loc))
        elif produc_str == 'NP -> NP VP':
            tree_loc = get_treeloc_from_producid(productions,produc_id)
            np_tree = parse_tree[tree_loc]
            target, pos, loc = parse_vp(np_tree)
            locs.append((target, pos, loc))
        elif produc_str == 'VP -> VB NP PP' or produc_str == 'VP -> VB NP PP PP':
            tree_loc = get_treeloc_from_producid(productions,produc_id)
            vp_tree = parse_tree[tree_loc]
            target, pos, loc = parse_nppp(vp_tree)
            locs.append((target, pos, loc))
        else:
            pass
        # 将loc进行再一次的解析
    loc_info = parse_locs_norm(locs)
    return loc_info

def get_loc_info_sentence(parse_dict,sentence):
    '''
    从句子中获得位置信息，注意：这里的句子一定要是简单句，并且格式一定要规范，否则可能匹配不出来
    parse_dict:存放解析结果的dict
    return loc_info dict{target:locs[]}
    '''
    if sentence in parse_dict:
        parse_string = parse_dict[sentence]
        loc_info = get_loc_info_parse_string(parse_string)
    else:
        print(f'{sentence} has no parse result')
        loc_info = {}
    return loc_info

def union_loc_infos(loc_infos:list):
    '''
    将多个loc_infos结合起来
    '''
    loc_info_union = {}
    for loc_info in loc_infos:
        for target, locs in loc_info.items():
            if target in loc_info_union:
                for loc in locs:
                    if loc not in loc_info_union[target]:
                        loc_info_union[target].append(loc)
            else:
                # loc_info_union[target] = locs #NOTE 对list直接赋值容易出现混乱，这个name指向那个变量，之后修改这个name的变量的时候，这个变量也会被修改
                loc_info_union[target] = [loc for loc in locs]
    return loc_info_union

# def get_loc_info_low(parse_dict_low,lan_feature):
#     '''
#     从low level instruction 里面读取位置信息
#     lan_feature：从traj里面读取的low level instruction
#     '''
#     lan_feature = rewrite_sentence(lan_feature,'low')
#     sentences = lan_feature.split(".")
#     obj_loc_infos = []
#     for sub_sentence in sentences:
#         sub_sentence = sub_sentence.strip()
#         if sub_sentence == '':
#             continue
#         if sub_sentence in parse_dict_low:
#             loc_info = get_loc_info_sentence(parse_dict_low,sub_sentence)
#             obj_loc_infos.append(loc_info)
#     # 将所有信息合并起来
#     loc_info_union = union_loc_infos(obj_loc_infos)
#     return loc_info_union
def get_loc_info(parse_dict,lan_feature,lan_granularity):
    '''
    从low level instruction 里面读取位置信息
    lan_feature：从traj里面读取的low level instruction
    '''
    lan_feature = rewrite_sentence(lan_feature,lan_granularity)
    sentences = lan_feature.split(".")
    obj_loc_infos = []
    for sub_sentence in sentences:
        sub_sentence = sub_sentence.strip()
        if sub_sentence == '':
            continue
        if sub_sentence in parse_dict:
            loc_info = get_loc_info_sentence(parse_dict,sub_sentence)
            obj_loc_infos.append(loc_info)
        else:
            print(f'no match parse result for {lan_feature}')
    # 将所有信息合并起来
    loc_info_union = union_loc_infos(obj_loc_infos)
    return loc_info_union  

# ***************
# def get_target_loc(parse_tree):
#     '''
#     从解析后的句法树中获得target(要找的东西)，p_word方位词，loc位置
#     有如下模板：
#     1. pick up the apple on the table (vp(np(nn pp)))
#     2. pick the apple up from the table (vp np(nn) pp),这类模板还有 put the apple on the table, 因此需要判断一下
#     '''
#     # 或许需要先判断一下句式？
#     for sub_tree in parse_tree.subtrees():
#         target = None
#         pos_word = None
#         loc = None
#         if sub_tree.label() == 'NP':
#             np_tree = sub_tree
#             for sub_tree in np_tree.subtrees():
#                 if (sub_tree.label() == 'NN' or sub_tree.label()=='NNS')and target is None :
#                     target = sub_tree.leaves()[0]
#                 if sub_tree.label() == 'PP':
#                     pp_tree = sub_tree
#                     for sub_tree in pp_tree.subtrees():
#                         if (sub_tree.label() == 'IN' or sub_tree.label()=='VBN' )and pos_word is None:
#                             pos_word = sub_tree.leaves()[0]
#                         if (sub_tree.label() == 'NN' or sub_tree.label()=='NNS')and loc is None:
#                             loc = sub_tree.leaves()[0]
#                         if pos_word is not None and loc is not None:
#                             break
#                 if target is not None and pos_word is not None and loc is not None:
#                     break
#             if target is not None and pos_word is not None and loc is not None:
#                 break
#     if target is None or pos_word is None or loc is None:
#         # 可能是另一种句式
#         for sub_tree in parse_tree.subtrees():
#             target = None
#             pos_word = None
#             loc = None
#             # 从另一个短语里面找，应该把所有东西都置为None
#             if sub_tree.label() == 'VP':
#                 vp_tree = sub_tree
#                 for sub_tree in vp_tree.subtrees():
#                     if sub_tree.label() == 'VB':
#                         verb = sub_tree.leaves()[0]
#                         if verb not in pick_verb:
#                             break
#                     if sub_tree.label() == 'NP':
#                         np_tree = sub_tree
#                         for sub_tree in np_tree.subtrees():
#                             if (sub_tree.label() == 'NN' or sub_tree.label()=='NNS') and target is None:
#                                 target = sub_tree.leaves()[0]
#                                 break
#                     if sub_tree.label() == 'PP':
#                         pp_tree = sub_tree
#                         for sub_tree in pp_tree.subtrees():
#                             if (sub_tree.label() == 'IN' or sub_tree.label()=='VBN') and pos_word is None:
#                                 pos_word = sub_tree.leaves()[0]
#                             if (sub_tree.label() == 'NN' or sub_tree.label()=='NNS') and loc is None:
#                                 loc = sub_tree.leaves()[0]
#                             if pos_word is not None and loc is not None:
#                                 break
#                     if target is not None and pos_word is not None and loc is not None:
#                         break
#                 if target is not None and pos_word is not None and loc is not None:
#                     break               
#     return target, pos_word, loc
# 此方法没有使用句子的productions来解析位置，已弃用
# 但是从结果来看，两者似乎没有太大的区别


# ********************
# 不通过句法分析来匹配的方法：仅仅检测大物体词汇，或者仅仅通过触发词来提取词汇
def get_loc_loose(sentence:list):
    '''
    从句子中获得location,使用比较宽松的准则：只要有名词相匹配就提取
    TODO 后期可能需要将名词提取出来，再根据场景中的物体，映射一下？
    '''
    prob_locs = []
    for word in sentence:
        word = word.lower().strip()
        if word not in OPENABLE_CLASS_LOWER:
            alfred_name = noun_map(word) 
            if alfred_name is not None and alfred_name in constants.FILM_LARGE_OBJ:
                prob_locs.append(alfred_name)
    return prob_locs

def get_loc_strict(sentence:list):
    ''''
    从句子中获得位置，使用比较严格的准则，只有遇到关键词才提取位置：关键词：from，on,
    '''
    trigger_words = ['from','on']
    match_switch = False
    prob_locs = []
    for word in sentence:
        word = word.lower().strip()
        if word in trigger_words:
            match_switch = True
        if match_switch:
            alfred_name = noun_map(word) 
            if alfred_name is not None and alfred_name in constants.FILM_LARGE_OBJ:
                prob_locs.append(alfred_name)
    return prob_locs
# *************************

# **************
# 最终提供给上层调用的函数
def parse_loc_high(high_language,match_method,target=None,parse_dict_high = None):
    '''
    从high languge 里面解析物体位置
    match_method:'loose','strict','strict_loose','parse'
    '''
    if match_method =='loose':
        prob_locs = get_loc_loose(high_language)
    elif match_method =='strict':
        prob_locs = get_loc_strict(high_language)
    elif match_method == 'strict_loose':
        prob_locs = get_loc_strict(high_language)
        prob_locs_loose = get_loc_loose(high_language)
        for loc in prob_locs_loose:
            if loc not in prob_locs:
                prob_locs.append(loc)
    elif match_method == 'parse':
        loc_info = get_loc_info(parse_dict_high, high_language,'high')
        if target in loc_info:
            prob_locs = loc_info[target]
        else:
            prob_locs = []
    return prob_locs

def parse_loc_low(low_language,match_method,target=None,parse_dict_low = None):
    '''
    从low level language 里面解析物体位置
    '''
    if match_method in ['strict','loose','strict_loose']:
        prob_locs = []
        for sub_instr in low_language:
            prob_locs += parse_loc_high(sub_instr,match_method)

        # 去除重复的
        pro_locs_drop_duplicate = []
        for item in prob_locs:
            if item not in pro_locs_drop_duplicate:
                pro_locs_drop_duplicate.append(item)
    else:
        assert target is not None, 'target must not be None in parse method'
        assert parse_dict_low is not None, 'parse_dict_low must not be None in parse method'
        loc_info = get_loc_info(parse_dict_low, low_language,'low')
        if target in loc_info:
            pro_locs_drop_duplicate = loc_info[target]
        else:
            pro_locs_drop_duplicate = []
    # print(pro_locs_drop_duplicate)
    return pro_locs_drop_duplicate
# ****************