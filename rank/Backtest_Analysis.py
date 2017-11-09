import src.backtest_percent as backtest_percent
import src.backtest_fix as backtest_fix
import sys
import argparse

parser = argparse.ArgumentParser(description='This is a backtest script')
parser.add_argument('-f','--read', help='read file',required=True)
parser.add_argument('-p','--input', help='input period',required=True)
parser.add_argument('-n','--port', help='input protfolio',required=True)
parser.add_argument('-o','--output', help='output file name')
parser.add_argument('-m','--method', help='method: percent or fix number', required=True)
args = parser.parse_args()

if args.method == 'percent': 
 result_percent = backtest_percent.backtest(args.read, args.input, args.port)
 backtest_percent.output(result_percent, args.read, args.output)
elif args.method == 'fix':
 result_fix = backtest_fix.backtest(args.read, args.input, args.port)
 backtest_fix.output(result_fix, args.read, args.output)