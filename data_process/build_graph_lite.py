# coding=utf-8
import json
import os
from collections import defaultdict

Item_Feature = '../data/ECommAI_ubp_round1_item_feature'
User_Feature = '../data/ECommAI_ubp_round1_user_feature'
User_Item = '../data/ECommAI_ubp_round1_train'
Test = '../data/ECommAI_ubp_round1_test'

User_Lookup = 'user_lookup'
Item_Lookup = 'item_lookup'
Test_Index = 'test_index'
Index_Train = 'index_train'
Graph_Saved = 'graph_json'
Degree_Less = 'degree_less_item'
Out_Path = '../data'

behavior = {'buy': '0', 'cart': '1', 'clk': '2', 'collect': '3'}


def map_line_u(line):
    # fields: 0:user id 1:item id 2: behavior 3: date
    fields = line.split()
    return int(fields[0]), (int(fields[1]), behavior[fields[2]])


def map_line_i(line):
    fields = line.split()
    return int(fields[1]), (int(fields[0]), behavior[fields[2]])


def remove_same_edge(line):
    out = []
    s = set()
    for edge in line:
        s.add(str(edge[0])+' '+edge[1])
    for edge in s:
        t = edge.split()
        out.append((int(t[0]), t[1]))
    return out


def to_block(k, v):
    block = {
        'node_id': k,
        'node_weight': 1.0,
        'uint64_feature': {},
        'float_feature': {},
        'binary_feature': {},
        'neighbor': {},
        'edge': []
    }
    for p in v:
        if p[1] not in block['neighbor']:
            block['neighbor'][p[1]] = {}
        block['neighbor'][p[1]][str(p[0])] = 1.0
        edge = {
            'src_id': k,
            'dst_id': p[0],
            'edge_type': int(p[1]),
            'weight': 1.0,
            'uint64_feature': {},
            'float_feature': {},
            'binary_feature': {}
        }
        block['edge'].append(edge)
    return block


def to_block_user(pair):
    block = to_block(pair[0], pair[1])
    block['node_type'] = 0
    return json.dumps(block)


def to_block_item(pair):
    block = to_block(pair[0], pair[1])
    block['node_type'] = 1
    return json.dumps(block)


def find_item_degree_less(lines, degree=10):
    item_degree = defaultdict(int)
    with open(os.path.join(Out_Path, Degree_Less), "w") as data:
        for line in lines:
            values = line.strip().split("\t")
            item_degree[values[1]] += 1
        for item in item_degree:
            if item_degree[item] > 10:
                write_str = item + "\n"
                data.write(write_str)
    '''
    lines.map(lambda line: (int(line.split()[1]), 1)).reduceByKey(lambda x, y: x + y).\
        filter(lambda x: x[1] <= degree).map(lambda x: x[0]).coalesce(1, True).\
        saveAsTextFile(os.path.join(Out_Path, Degree_Less))
    '''


def generate_lite(degree_less_item_file):
    user_dict = {}
    item_dict = {}
    dl_item = set()
    index = 0
    with open(degree_less_item_file, 'r') as df:
        for line in df:
            dl_item.add(int(line))

    print('生成User Index...')
    with open(User_Feature, 'r') as f_u, open(os.path.join(Out_Path, User_Lookup), 'w') as o_f_u:
        for line in f_u:
            uid = int(line.split()[0])
            user_dict[uid] = index
            o_f_u.write('%d\t%d\n' % (index, uid))
            index += 1
        # 找出在测试文件中出现，但在用户资料中不存在的记录
        with open(Test, 'r') as o_f_tu, open(os.path.join(Out_Path, Test_Index), 'w') as o_f_test:
            for line in o_f_tu:
                tuid = int(line.split()[0])
                if tuid not in user_dict:
                    user_dict[tuid] = index
                    o_f_u.write('%d\t%d\n' % (index, tuid))
                    index += 1
                    print('Out Range User: %d' % tuid)
                o_f_test.write('%d\n' % user_dict[tuid])
    print('生成Item Index...')
    with open(Item_Feature, 'r') as f_i, open(os.path.join(Out_Path, Item_Lookup), 'w') as o_f_i:
        for line in f_i:
            iid = int(line.split()[0])
            if iid not in dl_item:
                item_dict[iid] = index
                o_f_i.write('%d\t%d\n' % (index, iid))
                index += 1
    print('生成User-Item...')
    with open(User_Item, 'r') as f_ui, open(os.path.join(Out_Path, Index_Train), 'w') as o_f_ui:
        for line in f_ui:
            fields = line.split()
            uid, iid = int(fields[0]), int(fields[1])
            if uid not in user_dict: continue
            if iid not in item_dict: continue
            fields[0] = str(user_dict[uid])
            fields[1] = str(item_dict[iid])
            o_f_ui.write('\t'.join(fields) + '\n')


# 构图

def build_graph_json():
    lines = open(os.path.join(Out_Path, Index_Train), "r")
    test_uids = set()
    with open(os.path.join(Out_Path, Test_Index), "r") as test_uid_data:
        for value in test_uid_data:
            test_uid = value.strip().split()[0]
            test_uids.add(test_uid)
    train_uids = set()
    for value in lines:
        train_uid = value.strip().split()[0]
        train_uids.add(train_uid)
    print("train uids is: ", len(train_uids))
    cs_uid = test_uids.intersection(train_uids)
    print("create u_i")
    u_i = []
    i_u = []
    lines = open(os.path.join(Out_Path, Index_Train), "r")
    for line in lines:
        user = map_line_u(line)
        u_i.append(user)
        item = map_line_i(line)
        i_u.append(item)
    #u_i = map(map_line_u, lines)

    u_is = defaultdict(list)
    for (key, value) in u_i:
        u_is[key].append(value)
    for key in u_is:
        remove_edges = remove_same_edge(u_is[key])
        u_is[key] = remove_edges

    i_us = defaultdict(list)
    for (key, value) in i_u:
        i_us[key].append(value)
    for key in i_us:
        remove_edges = remove_same_edge(i_us[key])
        i_us[key] = remove_edges
    top_k_items = sorted(i_us.items(), key=lambda item: len(item[1]), reverse=True)
    s_i_top = map(lambda x: x[0], top_k_items[:10])

    for uid in cs_uid:
        items = []
        for item in s_i_top:
            items.append((item, 2))
        u_is[uid] = items
    for nid in s_i_top:
        users = []
        for user in cs_uid:
            users.append((user, 2))
        i_us[nid].extend(users)

    u_is_items = u_is.items()
    u_block = map(to_block_user, u_is_items)

    i_us_items = i_us.items()
    i_block = map(to_block_item, i_us_items)

    result_path = open(os.path.join(Out_Path, Graph_Saved), "w")
    for user_node in u_block:
        result_path.write(user_node + "\n")
    for item_node in i_block:
        result_path.write(item_node + "\n")





def build_graph_json_origin(lines):

    test_uid = open(os.path.join(Out_Path, Test_Index), "r")

    # 找出测试集中的冷启动用户
    cs_uid = test_uid.subtract(lines.map(lambda l: l.split()[0])).map(lambda x: int(x))

    u_i = lines.map(map_line_u)
    i_u = lines.map(map_line_i)

    u_is = u_i.combineByKey(lambda x: [x], lambda x, y: x + [y], lambda x, y: x + y).mapValues(remove_same_edge)
    i_us = i_u.combineByKey(lambda x: [x], lambda x, y: x + [y], lambda x, y: x + y).mapValues(remove_same_edge)

    # 找出度最大的Top10物品
    s_i_top = sc.parallelize(i_us.map(lambda x: (len(x[1]), x[0])).sortByKey(False).top(10)).\
        map(lambda x: x[1])

    # 冷启动节点和Top10物品连接点击边
    u_o_is = cs_uid.cartesian(s_i_top).mapValues(lambda x: (x, '2')).\
        combineByKey(lambda x: [x], lambda x, y: x + [y], lambda x, y: x + y)
    # 反向连接
    i_o_us = s_i_top.cartesian(cs_uid).mapValues(lambda x: (x, '2')).\
        combineByKey(lambda x: [x], lambda x, y: x + [y], lambda x, y: x + y)

    # 合并
    u_is = u_is.union(u_o_is)
    i_us = i_us.union(i_o_us).reduceByKey(lambda x, y: x+y)

    u_block = u_is.map(to_block_user)
    i_block = i_us.map(to_block_item)

    all_block = u_block.union(i_block)

    all_block.coalesce(10, True).saveAsTextFile(os.path.join(Out_Path, Graph_Saved))


# 统计节点度信息
def get_degree(lines):
    lines.flatMap(lambda l: l.split()[0:2]).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y).\
        map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x+y).sortByKey().\
        coalesce(1, True).saveAsTextFile(os.path.join(Out_Path, 'graph_degree'))


if __name__ == "__main__":
    item_degree = open(User_Item, "r")
    find_item_degree_less(item_degree)
    generate_lite(os.path.join(Out_Path, Degree_Less))
    build_graph_json()
    #print("User: %d" % lines.map(lambda l: l.split()[0]).distinct().count())
    #print("Item: %d" % lines.map(lambda l: l.split()[1]).distinct().count())
    # get_degree(lines)