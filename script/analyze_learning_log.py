# -*- coding: cp932 -*-
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
        record_pat = re.compile(r'^hirate eval = (?P<hirate_eval>.*) , dsig rmse = (?P<dsig_rmse>.*) , dsig mae = (?P<dsig_mae>.*) , eval mae = (?P<eval_mae>.*) , cross_entropy_eval = (?P<cee>.*) , cross_entropy_win = (?P<cew>.*)')
        log = []
        for line in fi.readlines():
            mo = sfens_pat.search(line)
            if mo:
                sfens = int(mo.groupdict()['sfens'])
                continue

            mo = record_pat.search(line)
            if mo:
                sfens += 1000000     # output every 1M sfens.
                if sfens < 10000000: # skip early period
                    continue;
                hirate_eval = float(mo.groupdict()['hirate_eval'])
                dsig_rmse    = float(mo.groupdict()['dsig_rmse'])
                dsig_mae    = float(mo.groupdict()['dsig_mae'])
                eval_mae    = float(mo.groupdict()['eval_mae'])
                cee         = float(mo.groupdict()['cee'])
                cew         = float(mo.groupdict()['cew'])
                log.append((sfens, hirate_eval, dsig_rmse , dsig_mae , eval_mae , cee , cew))

    if len(log) == 0:
        print('{}: Empty'.format(file_path))
        return None
    else:
        print('{}: {}'.format(file_path, len(log)))

    # dataframe
    df = pd.DataFrame(data=log, columns='sfens hirate_eval dsig_rmse dsig_mae eval_mae cee cew'.split())

    # plot
    fig, ax = plt.subplots(1, 1)
    ax.plot(
            df['sfens'],
            df['dsig_rmse'],
            '.-', color='blue', label='RMSE')
    ax.set_xlabel('# SFENs')
    ax.set_ylabel('DSIG_RMSE')
    ax.legend(loc='upper left').get_frame().set_alpha(0.5)

    ax = ax.twinx()
    ax.plot(
            df['sfens'],
            df['cee'],
		'.-', color='red', label='cee')
    ax.set_xlabel('# SFENs')
    ax.set_ylabel('cee')
    ax.legend(loc='upper right').get_frame().set_alpha(0.5)

    ax = ax.twinx()
    ax.plot(
            df['sfens'],
            df['cew'],
		'.-', color='green', label='cew')
    ax.set_xlabel('# SFENs')
    ax.set_ylabel('cew')
    ax.legend(loc='lower right').get_frame().set_alpha(0.5)

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
