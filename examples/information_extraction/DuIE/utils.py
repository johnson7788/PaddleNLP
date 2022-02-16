# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import json
import os
import re
import zipfile

import numpy as np


def find_entity(text_raw, id_, predictions, tok_to_orig_start_index,
                tok_to_orig_end_index):
    """
    在给定的predicate id下检索实体提及的某些预测。
    这是由 "解码 "函数调用的。
    :param text_raw:  '《只为不殇璃》是卡梅尼创作的网络小说，发表于晋江文学网'
    :type text_raw:
    :param id_: int : 3
    :type id_:
    :param predictions:  [[[0]], [[3]], [[1]], [[1]], [[1]], [[1]], [[0]], [[0]], [[58]], [[1]], [[1]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]]]
    :type predictions:
    :param tok_to_orig_start_index:  tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    :type tok_to_orig_start_index:
    :param tok_to_orig_end_index: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26])
    :type tok_to_orig_end_index:
    :return: 返回实体的文本 ['只为不殇璃']
    :rtype:
    """
    entity_list = []
    for i in range(len(predictions)):
        if [id_] in predictions[i]:
            j = 0
            while i + j + 1 < len(predictions):  # 找出O的位置，id_得到的是B的位置
                if [1] in predictions[i + j + 1]:
                    j += 1
                else:
                    break
            entity = ''.join(text_raw[tok_to_orig_start_index[i]:
                                      tok_to_orig_end_index[i + j] + 1])
            entity_list.append(entity)
    return list(set(entity_list))


def decoding(example_batch, id2spo, logits_batch, seq_len_batch,
             tok_to_orig_start_index_batch, tok_to_orig_end_index_batch):
    """
        模型输出到spo三元组格式
    model output logits -> formatted spo (as in data set file)
    :param example_batch:   原始数据, [batch_size, dict(text, spo_list)]
    :type example_batch:
    :param id2spo:  spo列表，里面包含predict关系，subject实体类型，object实体类型
    :type id2spo:
    :param logits_batch:  [batch_size, seq_len, num_labels]
    :type logits_batch:  模型预测的logits
    :param seq_len_batch:   每个样本的原始的长度 tensor([126,  61,  84,  39,  58,  48,  81, 105])
    :type seq_len_batch:
    :param tok_to_orig_start_index_batch: [batch_size, seq_len], 每个token的原始的开始位置
    :type tok_to_orig_start_index_batch:
    :param tok_to_orig_end_index_batch:  [batch_size, seq_len]， 每个token的原始的结束位置
    :type tok_to_orig_end_index_batch:
    :return:
    formatted_outputs = {list: 8} [{'text': '《乱舞春秋》 是周杰伦独树一帜的“中国风”歌曲，这首歌曲体现了周杰伦对传统文化的牢固把握，对中国元素的坚持运用，并用时尚的国外西方音乐手法表现中国古典情怀的经典作品', 'spo_list': []}, {'text': '安卓手机网创立于2011年1月，是国内领先的android资讯和资源提供商，创立之初，即秉承“用户第一”的经营理念，通过门户+论坛的方式，获得广大机友们的热爱安卓手机网目前提供，android资讯，手机报价，评测，安卓手机教程，论坛等，全方位的服务', 'spo_list': []}, {'text': '跨界喜剧王 第三季海一天化身办公室主任，文松直接变脸，求生欲超强', 'spo_list': []}, {'text': '基本简介《到不了的地方》作为高雄电影节长片周的闭幕片博得了一片赞扬声，《到不了的地方》由李鼎执导，林辰唏、林柏宏、张睿家、庹宗华等人主演，导演李鼎称该影片是金钟得奖的作品《装满的生活时光—延续自己的声音篇》的完整版，寄托了他对父亲的思念', 'spo_list': []}, {'text': '[音乐]岁月缝花--陈学冬', 'spo_list': []}, {'text': '小说娱乐之成功者系统1是作者哥是潇洒哥写的一本都市小说.', 'spo_list': []}, {'text': '纽约美食与创意论坛现场华美协进社1926年由约翰·杜威、孟禄和胡适、郭秉文等共同创建', 'spo_list': []}, {'text': '《只为不殇璃》是卡梅尼创作的网络小说，发表于晋江文学网', 'spo_list': [{'predicate': '作者', 'object_type': {'@value': '人物'}, 'subject_type': '图书作品', 'object': {'@value': '卡梅尼'}, 'subject': '只为不殇璃'}]}]
 0 = {dict: 2} {'text': '《乱舞春秋》 是周杰伦独树一帜的“中国风”歌曲，这首歌曲体现了周杰伦对传统文化的牢固把握，对中国元素的坚持运用，并用时尚的国外西方音乐手法表现中国古典情怀的经典作品', 'spo_list': []}
 1 = {dict: 2} {'text': '安卓手机网创立于2011年1月，是国内领先的android资讯和资源提供商，创立之初，即秉承“用户第一”的经营理念，通过门户+论坛的方式，获得广大机友们的热爱安卓手机网目前提供，android资讯，手机报价，评测，安卓手机教程，论坛等，全方位的服务', 'spo_list': []}
 2 = {dict: 2} {'text': '跨界喜剧王 第三季海一天化身办公室主任，文松直接变脸，求生欲超强', 'spo_list': []}
 3 = {dict: 2} {'text': '基本简介《到不了的地方》作为高雄电影节长片周的闭幕片博得了一片赞扬声，《到不了的地方》由李鼎执导，林辰唏、林柏宏、张睿家、庹宗华等人主演，导演李鼎称该影片是金钟得奖的作品《装满的生活时光—延续自己的声音篇》的完整版，寄托了他对父亲的思念', 'spo_list': []}
 4 = {dict: 2} {'text': '[音乐]岁月缝花--陈学冬', 'spo_list': []}
 5 = {dict: 2} {'text': '小说娱乐之成功者系统1是作者哥是潇洒哥写的一本都市小说.', 'spo_list': []}
 6 = {dict: 2} {'text': '纽约美食与创意论坛现场华美协进社1926年由约翰·杜威、孟禄和胡适、郭秉文等共同创建', 'spo_list': []}
 7 = {dict: 2} {'text': '《只为不殇璃》是卡梅尼创作的网络小说，发表于晋江文学网', 'spo_list': [{'predicate': '作者', 'object_type': {'@value': '人物'}, 'subject_type': '图书作品', 'object': {'@value': '卡梅尼'}, 'subject': '只为不殇璃'}]}
    :rtype:
    """
    formatted_outputs = []
    for (i, (example, logits, seq_len, tok_to_orig_start_index, tok_to_orig_end_index)) in \
            enumerate(zip(example_batch, logits_batch, seq_len_batch, tok_to_orig_start_index_batch, tok_to_orig_end_index_batch)):
        # 去掉CLS和SEP的logits
        logits = logits[1:seq_len + 1]  # slice between [CLS] and [SEP] to get valid logits
        # 概率大于0.5的为1，小于0.5的为0
        logits[logits >= 0.5] = 1
        logits[logits < 0.5] = 0
        tok_to_orig_start_index = tok_to_orig_start_index[1:seq_len + 1]          #获取原始的序列的长度
        tok_to_orig_end_index = tok_to_orig_end_index[1:seq_len + 1]
        predictions = []
        for token in logits:
            predictions.append(np.argwhere(token == 1).tolist())   # token是每个token的可能的标签，形状是[num_labels]

        # 将预测结果格式化为实例式输出
        formatted_instance = {}
        text_raw = example['text']   #eg: '《步步惊心》改编自著名作家桐华的同名清穿小说《甄嬛传》改编自流潋紫所著的同名小说电视剧《何以笙箫默》改编自顾漫同名小说《花千骨》改编自fresh果果同名小说《裸婚时代》是月影兰析创作的一部情感小说《琅琊榜》是根据海宴同名网络小说改编电视剧《宫锁心玉》，又名《宫》《雪豹》，该剧改编自网络小说《特战先驱》《我是特种兵》由红遍网络的小说《最后一颗子弹留给我》改编电视剧《来不及说我爱你》改编自匪我思存同名小说《来不及说我爱你》'
        complex_relation_label = [8, 10, 26, 32, 46]   # 写死了？
        complex_relation_affi_label = [9, 11, 27, 28, 29, 33, 47]

        # 扁平化预测，然后检索出所有有效的subject id
        flatten_predictions = []  # seq_len长度
        for layer_1 in predictions:
            for layer_2 in layer_1:
                if layer_2:
                    # 有的时候里面没有预测到位1的内容，都是为0的
                    flatten_predictions.append(layer_2[0])
                else:
                    flatten_predictions.append(0)
        subject_id_list = []
        for cls_label in list(set(flatten_predictions)):
            # 要预测出一对label，才放进subject_id_list中
            if 1 < cls_label <= 56 and (cls_label + 55) in flatten_predictions:
                subject_id_list.append(cls_label)
        subject_id_list = list(set(subject_id_list))  # subject_id_list可能为空，就是没预测到一对label， eg: [3]

        # fetch all valid spo by subject id
        spo_list = []
        for id_ in subject_id_list:
            if id_ in complex_relation_affi_label:
                continue  # do this in the next "else" branch
            if id_ not in complex_relation_label:
                subjects = find_entity(text_raw, id_, predictions,
                                       tok_to_orig_start_index,
                                       tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions,
                                      tok_to_orig_start_index,
                                      tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        spo_list.append({
                            "predicate": id2spo['predicate'][id_],   #id2spo['predicate'][id_]： '作者'
                            "object_type": {
                                '@value': id2spo['object_type'][id_]   #id2spo['object_type'][id_]： 人物
                            },
                            'subject_type': id2spo['subject_type'][id_],  # id2spo['subject_type'][id_]: 图书作品
                            "object": {
                                '@value': object_   #'卡梅尼'
                            },
                            "subject": subject_   # '只为不殇璃'
                        })
            else:
                #  traverse all complex relation and look through their corresponding affiliated objects
                subjects = find_entity(text_raw, id_, predictions,
                                       tok_to_orig_start_index,
                                       tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions,
                                      tok_to_orig_start_index,
                                      tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        object_dict = {'@value': object_}
                        object_type_dict = {
                            '@value': id2spo['object_type'][id_].split('_')[0]
                        }
                        if id_ in [8, 10, 32, 46
                                   ] and id_ + 1 in subject_id_list:
                            id_affi = id_ + 1
                            object_dict[id2spo['object_type'][id_affi].split(
                                '_')[1]] = find_entity(text_raw, id_affi + 55,
                                                       predictions,
                                                       tok_to_orig_start_index,
                                                       tok_to_orig_end_index)[0]
                            object_type_dict[id2spo['object_type'][
                                id_affi].split('_')[1]] = id2spo['object_type'][
                                    id_affi].split('_')[0]
                        elif id_ == 26:
                            for id_affi in [27, 28, 29]:
                                if id_affi in subject_id_list:
                                    object_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                    find_entity(text_raw, id_affi + 55, predictions, tok_to_orig_start_index, tok_to_orig_end_index)[0]
                                    object_type_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                    id2spo['object_type'][id_affi].split('_')[0]
                        spo_list.append({
                            "predicate": id2spo['predicate'][id_],
                            "object_type": object_type_dict,
                            "subject_type": id2spo['subject_type'][id_],
                            "object": object_dict,
                            "subject": subject_
                        })

        formatted_instance['text'] = example['text']
        formatted_instance['spo_list'] = spo_list
        formatted_outputs.append(formatted_instance)
    return formatted_outputs


def write_prediction_results(formatted_outputs, file_path):
    """write the prediction results"""

    with codecs.open(file_path, 'w', 'utf-8') as f:
        for formatted_instance in formatted_outputs:
            json_str = json.dumps(formatted_instance, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
        zipfile_path = file_path + '.zip'
        f = zipfile.ZipFile(zipfile_path, 'w', zipfile.ZIP_DEFLATED)
        f.write(file_path)

    return zipfile_path


def get_precision_recall_f1(golden_file, predict_file):
    r = os.popen(
        'python3 ./re_official_evaluation.py --golden_file={} --predict_file={}'.
        format(golden_file, predict_file))
    result = r.read()
    r.close()
    precision = float(
        re.search("\"precision\", \"value\":.*?}", result).group(0).lstrip(
            "\"precision\", \"value\":").rstrip("}"))
    recall = float(
        re.search("\"recall\", \"value\":.*?}", result).group(0).lstrip(
            "\"recall\", \"value\":").rstrip("}"))
    f1 = float(
        re.search("\"f1-score\", \"value\":.*?}", result).group(0).lstrip(
            "\"f1-score\", \"value\":").rstrip("}"))

    return precision, recall, f1
