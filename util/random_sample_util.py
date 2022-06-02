import os
import pandas as pd
import random
import util.file_util as ft
import logging
import argparse

# logging.basicConfig(filename='./sampling.log', level=logging.INFO)


def get_txt_to_df(path, src_lang:str):
    file_list = os.listdir(path)

    domain = []
    src_list = []
    tgt_list = []

    for file in file_list:
        if file.split('.')[1] == src_lang:
            file_name = file.split('_')[3]
            src_text_list = ft.get_all_lines(f"{path}/{file}")

            for src_text in src_text_list:
                src_list.append(src_text)
                domain.append(file_name)

        else:
            tgt_text_list = ft.get_all_lines(f"{path}/{file}")
            for tgt_text in tgt_text_list:
                tgt_list.append(tgt_text)

    df = pd.DataFrame({
        'domain' : domain,
        'src' : src_list,
        'tgt' : tgt_list
    })

    return df


def read_tab_file(input_path):
    df = pd.read_csv(input_path, sep='\t',error_bad_lines=False)
    df.columns = ['src', 'tgt']
    return df


def random_sampling(df, random_seed:int, sample_num:int, data:str):
    random.seed(random_seed)
    if sample_num > len(df):
        random_sample_index = random.sample(range(len(df)), len(df))
    else:
        random_sample_index = random.sample(range(len(df)), sample_num)

    if data == 'train':
        result_df = df.drop(random_sample_index)
        result_df.reset_index(inplace=True, drop=True)
        return result_df
    elif data == 'test':
        result_df = df.loc[random_sample_index]
        result_df.reset_index(inplace=True, drop=True)
        return result_df


def random_sampling_df(input_path, random_seed:int, sampling_num:int, data:str):
    df = read_tab_file(input_path)
    random.seed(random_seed)
    random_sample_index = random.sample(range(len(df)), sampling_num)

    if data == 'train':
        result_df = df.drop(random_sample_index)
        result_df.reset_index(inplace=True, drop=True)
        return result_df

    elif data == 'test':
        result_df = df.loc[random_sample_index]
        result_df.reset_index(inplace=True, drop=True)
        return result_df


def train_test_split_path(path, src_lang, random_seed, sampling_num, data):
    df = get_txt_to_df(path, src_lang)
    if data == 'train':
        return random_sampling(df, random_seed, sampling_num, data='train')
    elif data == 'test':
        return random_sampling(df, random_seed, sampling_num, data='test')


def domain_sample(input_path, random_seed,sample_num):
    '''

    :param sample_num: Extract Sample Count
    :return: Domain sample
    '''
    df = pd.read_csv(input_path)
    print(f'Data Frame Columns : {df.columns}')
    print(f'Data length : {len(df)}')
    if len(df) < sample_num:
        print('Merge Data Count smaller  than Sample Count---')

    else:
        domain_list = df['domain'].unique()
        d_sample = int(sample_num / len(domain_list))

        df_sample = pd.DataFrame()
        df_other = pd.DataFrame()
        for domain in domain_list:
            domain_extract = (df['domain'] == domain)
            domain_sample = df[domain_extract]
            domain_sample.reset_index(inplace=True, drop=True)
            print(f'{domain} Cnt is {len(domain_sample)}')
            domain_sample_df = random_sampling(domain_sample, random_seed=random_seed, sample_num=d_sample, data='test')
            df_sample = pd.concat([domain_sample_df, df_sample], axis=0)

            other = random_sampling(domain_sample, random_seed=random_seed, sample_num=d_sample, data='train')
            df_other = pd.concat([other, df_other], axis=0)

        print(f'Extracted Sample Data Count : {len(df_sample)}')
        df_sample.reset_index(inplace=True, drop=True)
        df_other.reset_index(inplace=True, drop=True)

        if len(df_sample) < sample_num:
            print(f'Extract additional sample data --- {sample_num - len(df_sample)}')
            sample_plus = random_sampling(df_other, random_seed=random_seed, sample_num=sample_num - len(df_sample),
                                              data='test')
            df_sample = pd.concat([sample_plus, df_sample], axis=0)
            return df_sample
        else:
            return df_sample


def args():
    parser = argparse.ArgumentParser(usage='usage', description='Usage of parameters ')
    parser.add_argument('--input', required=False, default='../data/kovi/20220427/raw/ko_en_vi_corpus_220427.xlsx')
    parser.add_argument('--output', required=False, default='../data/kovi/20220427/csv')
    return parser.parse_args()


def main():
    config = args()
    df = pd.read_excel(config.input)
    df = df[['ko', 'vi']]
    print(df.head)
    sample = random_sampling(df, random_seed=427, sample_num=1000, data='test')
    sample_other = random_sampling(df, random_seed=427, sample_num=1000, data='train')
    sample2 = random_sampling(sample_other, random_seed=2022, sample_num=1000, data='test')
    sample2.to_csv(f'{config.output}/confirm_sample1000_kovi2.csv', index=False, encoding='utf-8-sig')
    # print(f'End -- ')

if __name__ == '__main__':
    main()
