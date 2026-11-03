#!/usr/bin/python3
import yaml
import pexpect
import pexpect.popen_spawn
import pandas
import statsmodels.stats.weightstats
import cpuid
import re
import argparse
import platform
import logging

# python package install (Windows):
# https://www.microsoft.com/store/productId/9P7QFQMJRFP7
# python3 -m pip install cpuid pandas pexpect pyyaml statsmodels numpy==1.19.3

# python package install (Ubuntu):
# sudo apt-get update
# sudo apt-get install python3 python3-pip
# python3 -m pip install cpuid pandas pexpect pyyaml statsmodels numpy

# python package list:
# python3 -m pip list
# python3 -m pip list --outdated

pandas.options.display.float_format = '{:11.0f}'.format

parser = argparse.ArgumentParser(description='bench')
parser.add_argument('--cmd', dest='cmd', default='')
parser.add_argument('--loop', type=int, default=1)
parser.add_argument('--log', dest='log', default='bench.log')
parser.add_argument('engine1')
parser.add_argument('eval1')
parser.add_argument('engine2', default=None, nargs='?')
parser.add_argument('eval2', default=None, nargs='?')

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', filename=args.log, level=logging.DEBUG)
logger = logging.getLogger('bench')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info('OS: %s' % platform.system())
logger.info(yaml.dump(args).replace(r"!!python/object:argparse.Namespace", ''))

class YOBench():
  def __init__(self, path, eval, cmd):
    self.path = path
    self.eval = eval
    self.cmd = cmd

  def exec(self):
    rlines = ""
    child = pexpect.spawn(self.path) if platform.system() != "Windows" else pexpect.popen_spawn.PopenSpawn(self.path.replace("\\", "/"))
    child.sendline("setoption name EvalDir value %s" % self.eval)
    child.sendline("setoption name PvInterval value %s" % 0)
    child.sendline("isready")
    child.expect(r"readyok\s*\n", 60)
    rlines += child.before.decode("cp932", errors="ignore")
    rlines += child.after.decode("cp932", errors="ignore")
    child.sendline("bench %s" % self.cmd)
    child.expect(r"Nodes\/second +: \d+\s*\n", 600)
    rlines += child.before.decode("cp932", errors="ignore")
    rlines += child.after.decode("cp932", errors="ignore")
    child.sendline("quit")
    if platform.system() != "Windows":
      child.close()
    return rlines

ptn1 = re.compile(r'Total time \(ms\) +: (?P<time_e1>\d+)\s+Nodes searched +: (?P<nodes_e1>\d+)\s+Nodes\/second +: (?P<nps_e1>\d+)')
ptn2 = re.compile(r'Total time \(ms\) +: (?P<time_e2>\d+)\s+Nodes searched +: (?P<nodes_e2>\d+)\s+Nodes\/second +: (?P<nps_e2>\d+)')

bench1 = YOBench(args.engine1, args.eval1, args.cmd)
bench2 = YOBench(args.engine2, args.eval2, args.cmd)

dlist = []

print('run       base       test     diff')
for i in range(args.loop):
  print('{:3d} '.format(i + 1), end='', flush=True)
  dic = {}

  if args.engine1 is not None :
    logger.debug('run {:d} {:s}'.format(i + 1, args.engine1))
    res = bench1.exec()
    logger.debug(res)
    m = ptn1.search(res).groupdict()
    if m is not None :
      for k, v in m.items():
        m[k] = int(v)
      dic.update(m)
      print('{:10d} '.format(m['nps_e1']), end='', flush=True)

  if args.engine2 is not None :
    logger.debug('run {:d} {:s}'.format(i + 1, args.engine2))
    res = bench2.exec()
    logger.debug(res)
    m = ptn2.search(res).groupdict()
    if m is not None :
      for k, v in m.items():
        m[k] = int(v)
      dic.update(m)
      print('{:10d} '.format(m['nps_e2']), end='', flush=True)

  if 'nps_e2' in dic :
    dic['nps_diff'] = dic['nps_e2'] - dic['nps_e1']
    print('{:+8d}'.format(dic['nps_diff']), flush=True)

    logger.debug('run       base       test     diff')
    logger.debug('{:3d} {:10d} {:10d} {:+8d}'.format(
      i + 1,
      dic['nps_e1'],
      dic['nps_e2'],
      dic['nps_diff'],
    ))
  else:
    print()
    logger.debug('{:3d} {:10d}'.format(
      i + 1,
      dic['nps_e1']
    ))

  dlist.append(dic)

df = pandas.json_normalize(dlist)
df_mean = df.mean()
df_std = df.std()

if args.engine2 is not None :
  logger.info('''

{:s}

{:s}

Result of {:d} runs
==================
base           = {:10.0f} +/- {:.0f}
test           = {:10.0f} +/- {:.0f}
diff           = {:+10.0f} +/- {:.0f}

speedup        = {:+.4f}
P(speedup > 0) = {:.4f}

Vendor ID         : {:s}
CPU Name          : {:s}
Microarchitecture : {:s}

'''.format(
    str(df),
    str(df.describe()),
    args.loop,
    df_mean['nps_e1'], df_std['nps_e1'],
    df_mean['nps_e2'], df_std['nps_e2'],
    df_mean['nps_diff'], df_std['nps_diff'],
    df_mean['nps_diff'] / df_mean['nps_e1'],
    statsmodels.stats.weightstats.ttest_ind(
      df['nps_e1'].values,
      df['nps_e2'].values,
      alternative='larger',
      usevar='unequal'
    )[1],
    cpuid.cpu_vendor(),
    cpuid.cpu_name(),
    '%s%s' % cpuid.cpu_microarchitecture()
  ))
else:
  logger.info('''

{:s}

{:s}

Result of {:d} runs
==================
base           = {:10.0f} +/- {:.0f}

Vendor ID         : {:s}
CPU Name          : {:s}
Microarchitecture : {:s}

'''.format(
    str(df),
    str(df.describe()),
    args.loop,
    df_mean['nps_e1'], df_std['nps_e1'],
    cpuid.cpu_vendor(),
    cpuid.cpu_name(),
    '%s%s' % cpuid.cpu_microarchitecture()
  ))
