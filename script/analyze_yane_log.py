# -*- coding: utf8 -*-
import os
import sys
import re
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd


def analyze_log(file_path):
    with open(file_path, 'rb') as fi:
        sfens_pat = re.compile(r'^(?P<sfens>\d+) sfens ,')
        record_pat = re.compile(r'^rmse = (?P<rmse>.*) , mean_error = (?P<mean_error>.*)')
        # mini-batch size , added by yane.
        mini_batch_pat = re.compile(r'mini-batch size : (?P<mini_batch>\d+)')

        log = []
        sfen_counter = 0
        for line in fi.readlines():
            mo = sfens_pat.search(line)
            if mo:
                sfens = int(mo.groupdict()['sfens'])
                counter = 0
                continue

            mo = record_pat.search(line)
            if mo:
                rmse = float(mo.groupdict()['rmse'])
                mean_error = float(mo.groupdict()['mean_error'])
                counter += 1
                sfen_counter += 1

                # 最初の数回は値が発散するので無視する。
                if sfen_counter >= 5:
                    log.append((sfens, counter, rmse, mean_error))

				# mini-batch size , added by yane.
                sfens += mini_batch

            mo = mini_batch_pat.search(line)
            if mo:
                mini_batch = int(mo.groupdict()['mini_batch'])

    if len(log) == 0:
        print('{}: Empty'.format(file_path))
        return None
    else:
        print('{}: {}'.format(file_path, len(log)))

    # dataframe
    df = pd.DataFrame(data=log, columns='sfens counter rmse mean_error'.split())

    # plot
    fig, ax = plt.subplots(1, 1)
    ax.plot(
            df['sfens'],
            df['rmse'],
            '.-', color='blue', label='RMSE')
    ax.set_xlabel('# SFENs')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper left').get_frame().set_alpha(0.5)
    ax = ax.twinx()
    ax.plot(
            df['sfens'],
            df['mean_error'],
            '.-', color='red', label='mean_error')
    ax.set_xlabel('# SFENs')
    ax.set_ylabel('mean error')
    ax.legend(loc='upper right').get_frame().set_alpha(0.5)
    ax.set_title(file_path)

    return fig

if __name__ == '__main__':

    with matplotlib.backends.backend_pdf.PdfPages('yane.pdf') as pdf:
        for file_path in sorted(glob.glob(os.path.join(sys.argv[1], '*', 'log'))):
            fig = analyze_log(file_path)
            if fig is not None:
                pdf.savefig(fig)

        d = pdf.infodict()
        d['Title'] = u'Yanelog analysis of [{}]'.format(sys.argv[1])
        d['CreationDate'] = datetime.datetime.now()

    plt.show()
