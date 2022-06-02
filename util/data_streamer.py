import os
import pandas as pd
import util.file_util as ft
import argparse


class Data_Streamer:
    def __init__(self, src_lang, tgt_lang, save_path):
        '''
        Data_Streamer : read : 파일 확장자(sep, csv, tsv)별 Read
                            - extension : sep or csv or tsv
                            - read_path : file directory path
                        write : 확장자별 write
                            - extension : sep or csv or tsv

        :param src_lang: Source language
        :param tgt_lang: Target language
        :param save_path: save directory path
        '''
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.save_path = save_path

    def __repr__(self):
        return f'{self.src_lang}, {self.tgt_lang}, {self.save_path}'

    def read(self, extension='csv', read_path=''):
        if extension not in ['sep', 'csv', 'tsv']:
            print(f'Not Supported your extension: {extension}')

        print(f'Start merging with {extension}')
        print(f'input path : {read_path}')
        print(f'lang code: {self.src_lang}-{self.tgt_lang}')

        flist = os.listdir(read_path)
        concat_df = pd.DataFrame()
        if extension == 'tsv':
            for file in flist:
                if ft.get_extension(f'{read_path}/{file}') == 'tsv':
                    df = pd.read_csv(f'{read_path}/{file}', names=(['src', 'tgt']), sep='\t', header=None)
                    filename = file.split('.')[0]
                    df['filename'] = filename
                    df = df[['filename', 'src', 'tgt']]
                    if len(df['src']) != len(df['tgt']):
                        raise ValueError(f'Both sides are not equal: {file}')
                    concat_df = pd.concat([df, concat_df], axis=0)
                else:
                    return None
            concat_df.reset_index(inplace=True, drop=True)
            return concat_df

        elif extension == 'csv':
            concat_df = pd.DataFrame()
            for file in flist:
                if ft.get_extension(f'{read_path}/{file}') == 'csv':
                    df = pd.read_csv(f'{read_path}/{file}')
                    filename = file.split('.')[0]
                    df['filename'] = filename
                    df = df[['filename', 'src', 'tgt']]
                    if len(df['src']) != len(df['tgt']):
                        raise ValueError(f'Both sides are not equal: {file}')

                    concat_df = pd.concat([df, concat_df], axis=0)
                else:
                    return None
            concat_df.reset_index(inplace=True, drop=True)
            return concat_df

        elif extension == 'sep':
            src_file_list = [x for x in os.listdir(read_path) if x.endswith(self.src_lang)]
            tgt_file_list = [x for x in os.listdir(read_path) if x.endswith(self.tgt_lang)]
            src_file_list.sort()
            tgt_file_list.sort()

            concat_df = pd.DataFrame()
            for src_file, tgt_file in zip(src_file_list, tgt_file_list):
                src_list = ft.get_all_lines(read_path + '/' + src_file)
                tgt_list = ft.get_all_lines(read_path + '/' + tgt_file)
                print(f'size of text {ft.get_file_name_without_extension(src_file)}: {len(src_list)}-{len(tgt_list)}')
                if len(src_list) != len(tgt_list):
                    raise ValueError(f'Both sides are not equal: {src_file}-{tgt_file}')
                filename = src_file.split('.')[0]
                df = pd.DataFrame({'filename': filename, 'src': src_list, 'tgt': tgt_list})
                concat_df = pd.concat([df, concat_df], axis=0)
            concat_df.reset_index(inplace=True, drop=True)
            return concat_df
        else:
            print(f'Not support extension')
            return None

    def write(self, df, extension='csv'):
        if df is None:
            print(f'DataFrame is empty')
            return None

        if len(df['src']) != len(df['tgt']):
            raise ValueError(f'Both sides are not equal: {self.src_lang}-{self.tgt_lang}')
        else:
            print(f'Start save {extension} file --- ')
            if extension == 'csv':
                df.to_csv(f'{self.save_path}.csv', index=False, header=None)
                return None

            elif extension == 'tsv':
                text = [str(s) + '\t' + str(t) for s, t in zip(df['src'], df['tgt'])]
                ft.write_list_file(text, f'{self.save_path}.{self.src_lang}{self.tgt_lang}')
                return None

            elif extension == 'sep':
                src = [str(s) for s in df['src']]
                tgt = [str(s) for s in df['tgt']]
                if len(src) != len(tgt):
                    raise ValueError(f'Both sides are not equal: {self.src_lang}-{self.tgt_lang}')
                else:
                    ft.write_list_file(src, f'{self.save_path}.{self.src_lang}')
                    ft.write_list_file(tgt, f'{self.save_path}.{self.tgt_lang}')
                return None
            else:
                raise ValueError(f'Extension not supported: {extension}')


def define_args():
    parser = argparse.ArgumentParser(usage='usage', description='Usage of parameters ')

    parser.add_argument('--src', required=False, default='en', help='source LP')
    parser.add_argument('--tgt', required=False, default='ko', help='target LP')
    parser.add_argument('--req_type', required=False, default='sep', help='Request type')
    parser.add_argument('--res_type', required=False, default='tsv', help='Response type')
    parser.add_argument('--input', required=False,
                        default='../data/kovi/20220427/raw/ko_en_vi_corpus_220427.xlsx',
                        help='directory of files to be merged')
    parser.add_argument('--output', required=False,
                        default='../../faiss/data/ai-hub_medical',
                        help='directory to add merged files')
    return parser.parse_args()


def main():
    print('class merge_util read test ---')
    config = define_args()
    merge = Data_Streamer(config.src, config.tgt, config.output)
    print(merge)
    df = merge.read(extension='sep', read_path='../data/idko/train_makestar')
    print(df.head())
    df.columns = ['ko', 'src', 'tgt']
    merge.write(df, extension='tsv')


if __name__ == '__main__':
    main()